from rich.theme import Theme

# The Netra CLI Design System
# We use semantic names so we can reconfigure easily later.
NETRA_CLI_THEME = Theme(
    {
        "brand": "bold cyan",  # Logo, Main Titles
        "accent": "magenta",  # Key metrics that need to pop
        "success": "green",  # "Connected", "Done"
        "warning": "yellow",  # High Nulls, Skew
        "critical": "bold red",  # Errors, Crashes
        "muted": "dim white",  # Labels, Secondary Borders
        "value": "bold white",  # The actual data numbers
        "border.main": "dim cyan",  # The Outer Application Frame
        "border.section": "dim white",  # Inner section dividers
    }
)
