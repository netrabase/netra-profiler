from collections import defaultdict
from types import TracebackType
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from netra_profiler import __version__
from netra_profiler.cli.theme import NETRA_CLI_THEME
from netra_profiler.types import (
    ColumnMetrics,
    DiagnosticAlert,
    NetraProfile,
    PipelineContext,
    is_numeric_type,
)

LATENCY_THRESHOLD = 0.01
EXECUTION_TIME_THRESHOLD = 0.01
TOP_K_STRING_LENGTH = 12
ZERO_POINT_ONE = 0.1

console = Console(theme=NETRA_CLI_THEME)


class NetraCLIRenderer:
    """
    The UI Class for Netra Profiler CLI.

    Operates on a 'Stacked Card' layout philosophy within a live terminal display.
    The lifecycle moves through three distinct phases:
    1. Data Source Connection (Ingestion & Schema inference)
    2. Engine Status (Polars lazy execution & Telemetry display)
    3. Results Dashboard (Health Checks & Variable Explorer)
    """

    def __init__(self) -> None:
        # UI State Components

        self._data_source_name: str | None = None

        # Phase 1: The Data Source Connection Panel
        self._data_source_panel: RenderableType | None = None

        # Phase 2: The Engine Progress, Status & Telemetry Panels
        self._engine_progress_bar: Progress | None = None
        self._engine_status_panel: RenderableType | None = None
        self._engine_telemetry_panel: RenderableType | None = None

        # Phase 3: The Profiling Results Panels (Health + Variables)
        self._profiling_results: RenderableType | None = None

        # Phase 4: Pipeline Execution Context
        self._pipeline_info_panel: RenderableType | None = None

        # The Live Context
        self.live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=4,
            transient=False,  # Retain the final dashboard on the terminal after exit
            auto_refresh=True,
        )

    def _build_layout(self) -> Panel:
        """
        Dynamically builds the UI hierarchy.
        Only active (non-None) cards are stacked into the rendering group.
        """
        active_cards: list[RenderableType] = []

        if self._data_source_panel:
            active_cards.append(self._data_source_panel)

        if self._engine_status_panel:
            active_cards.append(self._engine_status_panel)
        if self._engine_telemetry_panel:
            active_cards.append(self._engine_telemetry_panel)

        if self._profiling_results:
            active_cards.append(self._profiling_results)

        if self._pipeline_info_panel:
            active_cards.append(self._pipeline_info_panel)

        # Main Outer Wrapper Title: Show filename if available, else show init state
        panel_title = (
            f"[not dim][brand]{self._data_source_name}[/brand][/]"
            if self._data_source_name
            else "[muted]Initializing...[/muted]"
        )

        # Wrap the active stacked cards in the main border
        return Panel(
            Group(*active_cards),
            title=panel_title,
            subtitle=f"[muted]Netra Profiler v{__version__}[/muted]",
            subtitle_align="right",
            border_style="border.main",
            box=box.ROUNDED,
            padding=(0, 1),  # top and bottom padding = 1
        )

    def _refresh(self) -> None:
        """Pushes the updated layout to the Live display."""
        self.live.update(self._build_layout())

    def __enter__(self) -> "NetraCLIRenderer":
        self.live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.live.stop()

    def render_fatal_error(self, step: str, message: str, hint: str = "") -> None:
        """
        Renders a semantic red error card and halts the visual progression.

        step: "data_source" or "profiling"
        """
        grid = Table.grid(expand=True)
        grid.add_column()

        step_title = step.replace("_", " ").title()

        # We use a simple bold header without ASCII boxes
        grid.add_row(f"[bold #FF004D]✖ {step_title} Error[/]")
        grid.add_row("")  # Blank line for readability
        grid.add_row(f"[critical]{message}[/critical]")

        if hint:
            grid.add_row("")  # Blank line for readability
            grid.add_row(f"[value]{hint}[/value]")

        error_renderable = Padding(grid, (1, 1))

        # Slot the error into the correct panel
        if step == "data_source":
            self._data_source_panel = error_renderable
        elif step == "profiling":
            self._engine_telemetry_panel = error_renderable
            self._engine_status_panel = None
            self._engine_progress_bar = None

        self._refresh()

    # --- Phase 1: Data Source Connection ---

    def render_data_source_spinner(self, data_source_name: str) -> None:
        """Initializes the UI with a spinner while loading the data file."""
        self._data_source_name = data_source_name

        grid = Table.grid(expand=True)
        grid.add_column()

        spinner = Spinner(
            name="arrow3", text=Text("Connecting to data stream...", style="muted"), style="brand"
        )
        # Optical alignment to ensure the spinner is flush with the success checkmark
        aligned_spinner = Padding(spinner, (0, 0, 0, 1))

        grid.add_row(aligned_spinner)

        self._data_source_panel = Panel(
            grid,
            title="[not dim][success]⛁[/success] [value]Data Source[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()

    def render_data_source_panel(
        self,
        file_info: dict[str, str],
        schema_info: str,
        columns: int,
        time_taken: float,
    ) -> None:
        """Transforms the active loader into a permanent Data Source metadata card."""
        # 1. Latency Formatting
        latency_string = "< 0.01s" if time_taken < LATENCY_THRESHOLD else f"{time_taken:.2f}s"

        # Extract packaged file info
        file_path = file_info.get("path", "Unknown")
        file_size = file_info.get("size", "0 B")
        file_type = file_info.get("type", "Unknown")

        # 2. Vertical Tree Layout
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        success_msg = (
            " [not dim][success]✔[/success][/] "
            "[not dim][success]Connected to data source:[/success][/] "
            f"[value]{file_path}[/value]"
        )
        grid.add_row(success_msg)
        grid.add_row(f" └─ Latency: [not dim][brand]{latency_string}[/brand][/]")
        grid.add_row(f" └─ Format:  {file_type}")
        grid.add_row(f" └─ Size:    {file_size}")
        grid.add_row(f" └─ Schema:  {columns} Columns ({schema_info})")

        self._data_source_panel = Panel(
            grid,
            title="[not dim][success]⛁[/success] [value]Data Source[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()

    # --- Phase 2: Data Profiling & Engine Telemetry ---

    def render_engine_status_panel(self) -> Progress:
        """Initializes the progress bar state for the profiling operation."""
        self._engine_progress_bar = Progress(
            SpinnerColumn(style="brand"),
            TextColumn("[brand]{task.description}"),
            BarColumn(
                bar_width=None, style="#00FFFF", complete_style="#00FFFF", pulse_style="#FFFF00"
            ),
            expand=True,
            auto_refresh=False,
            console=console,
        )

        self._engine_status_panel = Panel(
            self._engine_progress_bar,
            title="[not dim][#FFDF00]⚙[/#FFDF00] [value]Engine Status[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()
        return self._engine_progress_bar

    def render_engine_telemetry_panel(
        self, engine_time: float, throughput_gb_s: float, peak_ram_usage: float
    ) -> None:
        """Morphs the active engine status into a permanent hardware telemetry card."""
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        engine_time_string = (
            "< 0.01s" if engine_time < EXECUTION_TIME_THRESHOLD else f"{engine_time:.2f}s"
        )

        grid.add_row(f" Engine Time:  [not dim][brand]{engine_time_string}[/brand][/]")
        grid.add_row(f" Data Throughput: [not dim][brand]{throughput_gb_s:.2f} GB/s[/brand][/]")
        grid.add_row(
            f" Peak RAM Usage:  [not dim][brand]{peak_ram_usage:.1f} MB[/brand][/] "
            "[muted](Includes Arrow allocation & compute buffers)[/muted]"
        )

        self._engine_telemetry_panel = Panel(
            grid,
            title="[not dim][#FFDF00]⏣[/#FFDF00] [value]Engine Telemetry[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        # We disable the active status card so the telemetry card takes its place
        self._engine_status_panel = None
        self._engine_progress_bar = None

        self._refresh()

    # --- Phase 3: Profiling Results ---

    # Data Health Panel
    def _build_health_summary_text(self, alerts: list[DiagnosticAlert], row_count: int) -> str:
        """Calculates the aggregate anomaly counts and formats the summary header."""
        row_count_string = f"[muted] Rows Profiled:[/muted] [value]{row_count:,}[/value]"

        if not alerts:
            return row_count_string

        critical_alerts_count = sum(1 for alert in alerts if alert["level"] == "CRITICAL")
        warning_alerts_count = sum(1 for alert in alerts if alert["level"] == "WARNING")

        if critical_alerts_count == 0 and warning_alerts_count == 0:
            return row_count_string

        issues_count_parts = []
        if critical_alerts_count > 0:
            issues_count_parts.append(f"[bold #8F0000]{critical_alerts_count} CRITICAL[/]")
        if warning_alerts_count > 0:
            s = "S" if warning_alerts_count > 1 else ""
            issues_count_parts.append(f"[bold #997602]{warning_alerts_count} WARNING{s}[/]")

        joined_issues_count = "[muted],[/muted] ".join(issues_count_parts)
        issues_count_string = (
            f"[muted]Data Issues found: \\[[/muted]{joined_issues_count}[muted]][/muted]"
        )
        return f"{row_count_string} [muted]|[/muted] {issues_count_string}"

    def _get_alert_badge(self, alert_level: str, alert_type: str) -> str:
        """Returns the color-coded alert badge based on alert severity."""
        if alert_level == "CRITICAL":
            return f"[bold #8F0000]\\[{alert_type}][/]"
        elif alert_level == "WARNING":
            return f"[bold #997602]\\[{alert_type}][/]"
        return f"[bold #8C8C8C]\\[{alert_type}][/]"

    def _render_data_health_panel(self, alerts: list[DiagnosticAlert], row_count: int) -> Panel:
        """Renders the prioritized list of dataset anomalies, anchored by the row count."""
        health_summary_text = self._build_health_summary_text(alerts, row_count)

        grid = Table.grid(padding=(0, 0))
        grid.add_column(justify="left")

        if not alerts:
            grid.add_row(
                " [bold green][HEALTHY][/]"
                "[value] Dataset is healthy. No anomalies detected.[/value]",
            )
        else:
            # 1. We group the alerts by column name
            alerts_by_column: dict[str, list[DiagnosticAlert]] = defaultdict(list)
            for alert in alerts:
                alerts_by_column[alert["column_name"]].append(alert)

            # 2. We sorted the grouped list alphabetically
            sorted_columns = sorted(alerts_by_column.keys())

            alert_level_ranks = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}

            for i, column in enumerate(sorted_columns):
                # Add spacing between column blocks (but not before the first one)
                if i > 0:
                    grid.add_row("")

                # Render the Column Name Header
                grid.add_row(f" [not dim][brand]{column}[/brand][/]")

                column_alerts = alerts_by_column[column]
                # 3. Alerts per column is sorted by alert level
                column_alerts.sort(key=lambda x: alert_level_ranks.get(x["level"], 3))

                # 4. Alerts per column is finally grouped by alert type
                seen_types = set()

                for alert in column_alerts:
                    alert_type = alert["type"]
                    alert_type_for_display = alert_type.replace("_", " ")
                    if alert_type in seen_types:
                        continue
                    seen_types.add(alert_type)

                    # Gather all messages for this specific alert type
                    messages = [
                        alert["message"] for alert in column_alerts if alert["type"] == alert_type
                    ]

                    badge = self._get_alert_badge(alert["level"], alert_type_for_display)

                    grid.add_row(f" [muted]└─[/muted] {badge}")

                    for msg in messages:
                        # We use bullet points if there are multiple alerts to create a clean list
                        prefix = "• " if len(messages) > 1 else ""
                        grid.add_row(f"      [muted]{prefix}{msg}[/muted]")

        panel_content = Group(health_summary_text, Padding(grid, (1, 0, 0, 0)))

        return Panel(
            panel_content,
            title="[not dim][#FF004D]✚[/] [value]Data Health[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def _group_column_metrics(
        self, profile: NetraProfile
    ) -> tuple[dict[str, ColumnMetrics], dict[str, ColumnMetrics]]:
        """Separates the hierarchical schema columns into Numerics and Categoricals."""

        numerics: dict[str, ColumnMetrics] = {}
        categoricals: dict[str, ColumnMetrics] = {}

        for column_name, metrics in sorted(profile["columns"].items()):
            if is_numeric_type(metrics.get("data_type", "")):
                numerics[column_name] = metrics
            else:
                categoricals[column_name] = metrics

        return numerics, categoricals

    def _build_sparkline(self, histogram_data: list[dict[str, Any]]) -> str:
        """
        Transforms histogram dictionary list into a Unicode sparkline.
        Maps bin counts to the 8 standard block characters.
        """
        if not histogram_data:
            return ""

        # Extract counts (Polars eager .hist() returns 'count' in its struct)
        counts = [float(bin_data.get("count", 0)) for bin_data in histogram_data]
        max_count = max(counts) if counts else 0

        if max_count == 0:
            return "[muted]" + (" " * len(counts)) + "[/muted]"

        # 9 characters: Space, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, Full Block
        bars = "  ▂▃▄▅▆▇█"
        sparkline = ""

        for count in counts:
            # Normalize to a 0.0 - 1.0 scale, then map to the 0-8 index of our bars string
            normalized = count / max_count
            bar_index = int(normalized * 8)
            sparkline += bars[bar_index]

        return f"[#7B619C]{sparkline}[/#7B619C]"

    def _build_top_k_string(self, top_k_data: list[dict[str, Any]], row_count: int) -> str:
        """
        Formats the Top-K dictionary list into a dense, readable summary string.
        E.g., "NL (55%), DE (20%)..."
        """
        if not top_k_data or row_count == 0:
            return "[value]-[/value]"

        parts = []
        # We only take the top 3 to prevent terminal wrapping
        for item in top_k_data[:3]:
            raw_value = item.get("value")
            count = item.get("count", 0)

            # Handle nulls cleanly
            value_string = "null" if raw_value is None else str(raw_value)

            # We truncate aggressively to protect horizontal real estate
            if len(value_string) > TOP_K_STRING_LENGTH:
                value_string = value_string[:9] + "..."

            value_percentage = (count / row_count) * 100

            # Only show percentages if they are meaningful (> 0%)
            if value_percentage >= 1:
                parts.append(f"{value_string} ({value_percentage:.0f}%)")
            else:
                parts.append(f"{value_string} (<1%)")

        return "[value]" + ", ".join(parts) + "[/value]"

    def _format_number(self, value: Any) -> str:
        """Formats floats to 2 decimal places, leaves ints alone, handles Nones."""
        if value is None:
            return "[muted]-[/muted]"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _format_null_percentage(self, null_count: int, row_count: int) -> str:
        """Calculates and formats the missing values percentage string."""
        if null_count == 0 or row_count == 0:
            return "[value]0%[/]"

        null_percentage = (null_count / row_count) * 100

        # Prevent tiny fractions from rounding down to a confusing "0.0%"
        if null_percentage < ZERO_POINT_ONE:
            return "[#C75B5B]< 0.1%[/]"

        return f"[#C75B5B]{null_percentage:.1f}%[/]"

    def _render_numeric_table(
        self, numerics: dict[str, ColumnMetrics], row_count: int
    ) -> Table | None:
        """Renders the dedicated table for numeric variables."""
        if not numerics:
            return None

        table = Table(
            box=box.SIMPLE, expand=True, show_edge=False, header_style="muted", border_style="muted"
        )
        table.add_column("Column", style="brand", justify="left")
        table.add_column("Missing", justify="right", style="value")
        table.add_column("Distinct", justify="right", style="value")
        table.add_column("Min", justify="right", style="value")
        table.add_column("Mean", justify="right", style="value")
        table.add_column("Max", justify="right", style="value")
        table.add_column("Distribution", justify="left")

        for column, column_profile in numerics.items():
            column_name_display_string = (
                f"{column} [muted]\\[{column_profile.get('data_type', '')}][/muted]"
            )
            null_percentage_string = self._format_null_percentage(
                column_profile.get("null_count", 0), row_count
            )

            distinct_string = f"{column_profile.get('n_unique', 0):,}"

            min_value = self._format_number(column_profile.get("min"))
            mean_value = self._format_number(column_profile.get("mean"))
            max_value = self._format_number(column_profile.get("max"))

            sparkline = self._build_sparkline(column_profile.get("histogram", []))

            table.add_row(
                column_name_display_string,
                null_percentage_string,
                distinct_string,
                min_value,
                mean_value,
                max_value,
                sparkline,
            )

        return table

    def _render_categorical_table(
        self, categoricals: dict[str, ColumnMetrics], row_count: int
    ) -> Table | None:
        """Renders the dedicated table for string, boolean, and categorical variables."""
        if not categoricals:
            return None

        table = Table(
            box=box.SIMPLE, expand=True, show_edge=False, header_style="muted", border_style="muted"
        )
        table.add_column("Column", style="brand", justify="left")
        table.add_column("Missing", justify="right", style="value")
        table.add_column("Distinct", justify="right", style="value")
        table.add_column("Lengths (Min/Avg/Max)", justify="center", style="value")
        table.add_column("Top Values", justify="left", style="value")

        for column, column_profile in categoricals.items():
            column_name_display_string = (
                f"{column} [muted]\\[{column_profile.get('data_type', '')}][/muted]"
            )

            null_percentage_string = self._format_null_percentage(
                column_profile.get("null_count", 0), row_count
            )

            distinct_string = f"{column_profile.get('n_unique', 0):,}"

            min_length = column_profile.get("min_length")
            mean_length = column_profile.get("mean_length")
            max_length = column_profile.get("max_length")

            if min_length is not None and mean_length is not None and max_length is not None:
                lengths_string = f"{min_length} / {mean_length:.1f} / {max_length}"
            else:
                lengths_string = "-"

            # Generate Top-K String
            top_k_string = self._build_top_k_string(column_profile.get("top_k", []), row_count)

            table.add_row(
                column_name_display_string,
                null_percentage_string,
                distinct_string,
                lengths_string,
                top_k_string,
            )

        return table

    def _build_variable_explorer_panel(
        self,
        numerics: dict[str, ColumnMetrics],
        categoricals: dict[str, ColumnMetrics],
        row_count: int,
    ) -> Panel | None:
        """Fuses the numeric and categorical tables into a single Variable Explorer panel."""
        numeric_table = self._render_numeric_table(numerics, row_count)
        categorical_table = self._render_categorical_table(categoricals, row_count)

        if not numeric_table and not categorical_table:
            return None

        renderables: list[RenderableType] = []

        if numeric_table:
            renderables.append(
                Text.from_markup(
                    " [not dim][accent]#/±[/accent] [value]Numeric Variables[/value][/]"
                )
            )
            renderables.append("")  # Breathing room under the title
            renderables.append(numeric_table)

        if categorical_table:
            if numeric_table:
                renderables.append("")
                renderables.append("")  # Extra space to clearly separate the two major sections
            renderables.append(
                Text.from_markup(
                    " [not dim][accent]A/B[/accent] [value]Categorical Variables[/value][/]"
                )
            )
            renderables.append("")  # Breathing room under the title
            renderables.append(categorical_table)

        return Panel(
            Group(*renderables),
            title="[not dim][brand]⊞[/brand] [value]Variable Explorer[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def render_profiling_results(self, profile: NetraProfile) -> None:
        """Assembles the final profiling results from the profile payload."""

        row_count = profile["dataset"]["row_count"]
        alerts = profile["alerts"]

        numerics, categoricals = self._group_column_metrics(profile)

        health_card = self._render_data_health_panel(alerts, row_count)
        variable_explorer_card = self._build_variable_explorer_panel(
            numerics, categoricals, row_count
        )

        renderables: list[Any] = [health_card]

        if variable_explorer_card:
            renderables.append(variable_explorer_card)

        self._profiling_results = Group(*renderables)
        self._refresh()

    def render_pipeline_info(self, pipeline_context: PipelineContext) -> None:
        """
        Renders the Pipeline Info card indicating the CI/CD Quality Gate status.
        Dynamically shifts its border color based on the pass/fail state.
        """
        quality_gate_active = pipeline_context.get("quality_gate_active", False)
        status = pipeline_context.get("status", "PASSIVE")
        exit_code = pipeline_context.get("exit_code", 0)
        reason = pipeline_context.get("reason", "")

        fail_on_critical = pipeline_context.get("fail_on_critical", False)
        fail_on_warnings = pipeline_context.get("fail_on_warnings", False)

        active_flags = []
        if fail_on_critical:
            active_flags.append("--fail-on-critical")
        if fail_on_warnings:
            active_flags.append("--fail-on-warnings")
        flags_string = ", ".join(active_flags) if active_flags else "None"

        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        if not quality_gate_active:
            border_style = "border.section"
            title = "[not dim][muted]⎆[/] [value]Pipeline Info[/value][/]"
            quality_gate_mode = "Passive"
        elif status == "PASSED":
            border_style = "success"
            title = "[not dim][success]⎆[/] [value]Pipeline Info[/value][/]"
            quality_gate_mode = "Active"
        else:
            border_style = "#FF004D"  # Bold Red
            title = "[not dim][#FF004D]⎆[/] [value]Pipeline Info[/value][/]"
            quality_gate_mode = "Active"

        # Build the Card Content
        grid.add_row(f" Quality Gate:  [value]{quality_gate_mode}[/value]")

        if quality_gate_active:
            grid.add_row(f" Flags:         [value]{flags_string}[/value]")

        exit_color = "success" if exit_code == 0 else "#FF004D"
        status_word = "Success" if exit_code == 0 else "Halted"
        grid.add_row(
            f" Exit Status:   [{exit_color}][bold]{exit_code} ({status_word})[/bold][/{exit_color}]"
        )

        if reason:
            grid.add_row(f" Reason:        [value]{reason}[/value]")

        self._pipeline_info_panel = Panel(
            grid,
            title=title,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()
