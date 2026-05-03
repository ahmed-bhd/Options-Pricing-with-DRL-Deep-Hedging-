# 📁 10_reports - Report Generation

## Overview

This module generates comprehensive analysis reports from backtest results.

## Report Formats

| Format | Description |
|--------|-------------|
| **Markdown** | Plain text, easy to edit, good for GitHub |
| **HTML** | Styled web page, good for sharing |
| **PDF** | Print-ready (requires weasyprint) |

## Report Contents

| Section | Description |
|---------|-------------|
| Executive Summary | Overall findings and winner |
| Performance Summary | Full metrics table |
| Risk Metrics | VaR, CVaR, Sortino, Calmar |
| DRL vs Classical | Direct comparison |
| Transaction Cost Analysis | Cost efficiency |
| Visualizations | Embedded figures |
| Conclusions | Recommendations |

## Usage Example

```python
from src.reports import ReportGenerator

# Generate reports
generator = ReportGenerator(results, comparison_df)
generator.generate_all(output_dir="10_reports")