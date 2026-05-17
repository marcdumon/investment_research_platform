"""Data-quality module: separate simfin/ and stooq/ submodules.

Each domain follows the same shape:
    {domain}_rules.py    @register + REGISTRY of Rule dataclass instances
    {domain}_runner.py   run(skip_reviewed=True) -> findings DataFrame
    {domain}_inspect.py  inspect(...) -> renderable artefact (DataFrame / Figure)
    {domain}_reviews.py  add_review, load_reviews, load_reviews_df + TOML I/O

Import from the submodule explicitly:
    from irp.quality.simfin_runner import run
    from irp.quality.stooq_runner  import run as stooq_run
"""
from irp.quality.edgar import filing_url

__all__ = ['filing_url']
