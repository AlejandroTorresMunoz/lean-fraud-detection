"""ETL transform stages composed by build_sequences — each pure and unit-testable.

Kept apart from the orchestrator (build_sequences) and the extract step (download) so every
stage can be tested in isolation: features (causal feature engineering), split (strict
time-based split), encode (categorical encoding + train-only scaling).
"""
