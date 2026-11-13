"""
Shared node/edge schema constants for FinGraphQA graph construction.
Keep all string labels here so graph_builder/check/evaluation stay consistent.
"""

# Node types
NODE_PAGE = "page"
NODE_TABLE = "table"
NODE_TEXT = "text_sentence"
NODE_TABLE_ROW = "table_row"
NODE_TABLE_COLUMN = "table_column"
NODE_TABLE_CELL = "table_cell"
NODE_QUANTITY = "quantity"
NODE_PERIOD = "period"
NODE_UNIT = "unit"
NODE_SCALE = "scale"
NODE_CURRENCY = "currency"
NODE_METRIC = "metric"
NODE_FINANCIAL_FACT = "financial_fact"
NODE_QUESTION = "question"
NODE_OPERATION = "operation"

# Source structure relations
REL_CONTAINS_TEXT = "contains_text"
REL_CONTAINS_TABLE = "contains_table"
REL_CONTAINS_ROW = "contains_table_row"
REL_CONTAINS_COLUMN = "contains_table_column"
REL_HAS_CELL = "has_cell"
REL_BELONGS_TO_ROW = "belongs_to_row"
REL_BELONGS_TO_COLUMN = "belongs_to_column"
REL_NEXT_ROW = "next_row"
REL_NEXT_COLUMN = "next_column"
REL_ADJACENT_TEXT = "adjacent_text"

# Entity/fact relations
REL_HAS_QUANTITY = "has_quantity"
REL_APPEARS_IN = "appears_in"
REL_HAS_PERIOD = "has_period"
REL_HAS_UNIT = "has_unit"
REL_HAS_SCALE = "has_scale"
REL_HAS_CURRENCY = "has_currency"
REL_HAS_METRIC = "has_metric"
REL_MENTIONED_IN = "mentioned_in"
REL_SOURCE_CELL = "source_cell"
REL_SOURCE_TEXT = "source_text"
REL_SOURCE_EVIDENCE = "source_evidence"

# Question relations
REL_ASKS_METRIC = "asks_metric"
REL_ASKS_PERIOD = "asks_period"
REL_ASKS_UNIT = "asks_unit"
REL_ASKS_OPERATION = "asks_operation"
REL_CANDIDATE_OPERAND = "candidate_operand"

# Context/fact relations
REL_SAME_METRIC_AS = "same_metric_as"
REL_SAME_PERIOD_AS = "same_period_as"
REL_SAME_ROW_FACT = "same_row_fact"
REL_SAME_COLUMN_FACT = "same_column_fact"
REL_PREVIOUS_PERIOD_OF = "previous_period_of"
REL_LATER_OR_EQUAL_PERIOD = "later_or_equal_period"
REL_PART_OF_TOTAL = "part_of_total"
REL_GREATER_OR_EQUAL_QUANTITY = "greater_or_equal_quantity"

# Table-text bridge relations
REL_SUPPORTED_BY_TEXT = "supported_by_text"
REL_DESCRIBES_TABLE = "describes_table"
REL_TABLE_TEXT_BRIDGE = "table_text_bridge"

BRIDGE_RELS = {REL_SUPPORTED_BY_TEXT, REL_DESCRIBES_TABLE, REL_TABLE_TEXT_BRIDGE}
