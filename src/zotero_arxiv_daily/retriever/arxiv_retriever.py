from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
    # 不建议这里再设 10 次内部重试，否则会和下面的外层重试叠加，
    # 一次失败可能卡非常久。
    client = arxiv.Client(num_retries=3, delay_seconds=10)

    query = '+'.join(self.config.source.arxiv.category)
    include_cross_list = self.config.source.arxiv.get(
        "include_cross_list", False
    )

    # Get the latest papers from arXiv RSS feed
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")

    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    raw_papers = []

    allowed_announce_types = (
        {"new", "cross"} if include_cross_list else {"new"}
    )

    all_paper_ids = [
        item.id.removeprefix("oai:arXiv.org:")
        for item in feed.entries
        if item.get("arxiv_announce_type", "new")
        in allowed_announce_types
    ]

    if self.config.executor.debug:
        all_paper_ids = all_paper_ids[:10]

    bar = tqdm(total=len(all_paper_ids))

    # 每批仍然保持 20 篇，避免因为 batch 太小反而增加 API 请求数量
    batch_size = 20

    # 外层只尝试 3 次即可。
    # arxiv.Client 内部本身还会 retry。
    max_batch_retries = 3

    for i in range(0, len(all_paper_ids), batch_size):
        batch_ids = all_paper_ids[i:i + batch_size]
        batch_number = i // batch_size

        search = arxiv.Search(id_list=batch_ids)

        batch_success = False

        for attempt in range(max_batch_retries):
            try:
                batch = list(client.results(search))

                raw_papers.extend(batch)
                bar.update(len(batch))

                batch_success = True
                break

            except arxiv.HTTPError as exc:
                # 429: Too Many Requests
                # 503: arXiv temporarily unavailable
                if exc.status in (429, 503):
                    if attempt < max_batch_retries - 1:
                        wait = 30 * (attempt + 1)

                        logger.warning(
                            f"arXiv API HTTP {exc.status} on "
                            f"batch {batch_number}, "
                            f"retry {attempt + 1}/"
                            f"{max_batch_retries - 1} "
                            f"in {wait}s"
                        )

                        sleep(wait)
                        continue

                logger.warning(
                    f"arXiv batch {batch_number} failed "
                    f"with HTTP {exc.status}. "
                    f"Falling back to per-paper requests."
                )

                break

        # ---------------------------------------------------------
        # 批量查询失败：
        # 不再让整个 workflow 退出，而是逐篇请求
        # ---------------------------------------------------------
        if not batch_success:
            logger.warning(
                f"Falling back to per-paper requests "
                f"for batch {batch_number} "
                f"({len(batch_ids)} papers)"
            )

            for index, paper_id in enumerate(batch_ids):
                try:
                    single_search = arxiv.Search(
                        id_list=[paper_id]
                    )

                    result = list(
                        client.results(single_search)
                    )

                    raw_papers.extend(result)
                    bar.update(len(result))

                except arxiv.HTTPError as exc:
                    logger.warning(
                        f"Skipping arXiv paper "
                        f"{paper_id}: HTTP {exc.status}"
                    )

                    # 即使这一篇失败，也继续下一篇
                    bar.update(1)

                except Exception as exc:
                    logger.warning(
                        f"Skipping arXiv paper "
                        f"{paper_id}: {exc}"
                    )

                    bar.update(1)

                # 单篇请求之间主动限速
                if index + 1 < len(batch_ids):
                    sleep(2)

        # batch 与 batch 之间也主动限速
        if i + batch_size < len(all_paper_ids):
            sleep(5)

    bar.close()

    logger.info(
        f"Successfully retrieved "
        f"{len(raw_papers)}/"
        f"{len(all_paper_ids)} arXiv papers"
    )

    return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
