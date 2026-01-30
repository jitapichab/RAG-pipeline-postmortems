#!/usr/bin/env python3
"""
Convert JIRA issues to Markdown files with YAML frontmatter for RAG ingestion.

This script extracts postmortem/incident data from JIRA and creates structured
markdown files optimized for retrieval.
"""

import json
import re
from pathlib import Path
from datetime import datetime


def extract_severity(description: str) -> str:
    """Extract severity from description text."""
    if not description:
        return "Unknown"
    
    patterns = [
        r"\*\*Severity:\*\*\s*\n?\s*(S[0-4]\s*-\s*[^\n]+)",
        r"Severity:\s*(S[0-4]\s*-\s*[^\n]+)",
        r"(S1|S2|S3|S4)\s*-\s*(Critical|High|Medium|Low)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Unknown"


def extract_field(description: str, field_name: str) -> str:
    """Extract a field value from description text."""
    if not description:
        return ""
    
    # Try markdown bold format: **Field:** value
    pattern = rf"\*\*{field_name}:\*\*\s*\n?\s*([^\n*]+)"
    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try plain format: Field: value
    pattern = rf"{field_name}:\s*([^\n]+)"
    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""


def extract_services(description: str) -> list:
    """Extract affected services from description."""
    services_str = extract_field(description, "Affected Services")
    if not services_str:
        return []
    
    # Split by comma, and, or newlines
    services = re.split(r'[,\n]|\s+and\s+', services_str)
    return [s.strip() for s in services if s.strip()]


def extract_root_cause_category(description: str) -> str:
    """Infer root cause category from description."""
    if not description:
        return "unknown"
    
    desc_lower = description.lower()
    
    categories = {
        "database": ["mongodb", "postgres", "aurora", "database", "index", "query", "deadlock"],
        "capacity": ["capacity", "load", "traffic", "volume", "scale", "pool", "exhaustion"],
        "configuration": ["config", "configuration", "setting", "parameter", "misconfigur"],
        "network": ["network", "dns", "timeout", "connection", "latency", "connectivity"],
        "deployment": ["deploy", "rollout", "release", "version", "migration"],
        "code_bug": ["bug", "logic", "null", "exception", "error handling", "missing filter"],
        "third_party": ["provider", "vendor", "external", "integration", "api"],
        "kafka": ["kafka", "consumer", "producer", "lag", "rebalance", "offset"],
        "redis": ["redis", "cache", "memory"],
    }
    
    for category, keywords in categories.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    
    return "unknown"


def sanitize_filename(text: str) -> str:
    """Create a safe filename from text."""
    # Remove special characters, keep alphanumeric, hyphens, underscores
    safe = re.sub(r'[^\w\s-]', '', text)
    safe = re.sub(r'\s+', '-', safe)
    return safe[:80].lower().strip('-')


def convert_issue_to_markdown(issue: dict) -> tuple[str, str]:
    """
    Convert a JIRA issue to markdown with frontmatter.
    
    Returns: (filename, content)
    """
    fields = issue.get("fields", {})
    key = issue.get("key", "UNKNOWN")
    summary = fields.get("summary", "No summary")
    description = fields.get("description", "")
    
    # Extract metadata
    created = fields.get("created", "")[:10]  # YYYY-MM-DD
    updated = fields.get("updated", "")[:10]
    
    status = fields.get("status", {}).get("name", "Unknown")
    priority = fields.get("priority", {}).get("name", "Unknown")
    issue_type = fields.get("issuetype", {}).get("name", "Task")
    
    labels = fields.get("labels", [])
    components = [c.get("name", "") for c in fields.get("components", []) if c.get("name")]
    
    assignee = fields.get("assignee", {})
    assignee_name = assignee.get("displayName", "Unassigned") if assignee else "Unassigned"
    
    reporter = fields.get("reporter", {})
    reporter_name = reporter.get("displayName", "Unknown") if reporter else "Unknown"
    
    # Extract from description
    severity = extract_severity(description)
    services = extract_services(description)
    owner_team = extract_field(description, "Owner")
    affected_clients = extract_field(description, "Affected Clientes") or extract_field(description, "Affected Clients")
    root_cause_category = extract_root_cause_category(description)
    
    # Extract incident date from description or summary
    incident_date = extract_field(description, "Date") or created
    
    # Build frontmatter
    frontmatter_lines = [
        "---",
        f'incident_id: "{key}"',
        f'title: "{summary.replace('"', '\\"')}"',
        f'severity: "{severity}"',
        f'date: "{incident_date}"',
        f'created: "{created}"',
        f'updated: "{updated}"',
        f'status: "{status}"',
        f'priority: "{priority}"',
        f'issue_type: "{issue_type}"',
        f'owner_team: "{owner_team}"',
        f'assignee: "{assignee_name}"',
        f'reporter: "{reporter_name}"',
        f'root_cause_category: "{root_cause_category}"',
    ]
    
    if services:
        services_yaml = json.dumps(services)
        frontmatter_lines.append(f'services_affected: {services_yaml}')
    else:
        frontmatter_lines.append('services_affected: []')
    
    if affected_clients:
        frontmatter_lines.append(f'affected_clients: "{affected_clients}"')
    
    if labels:
        labels_yaml = json.dumps(labels)
        frontmatter_lines.append(f'labels: {labels_yaml}')
    else:
        frontmatter_lines.append('labels: []')
    
    if components:
        components_yaml = json.dumps(components)
        frontmatter_lines.append(f'components: {components_yaml}')
    
    frontmatter_lines.append(f'jira_url: "https://yunopayments.atlassian.net/browse/{key}"')
    frontmatter_lines.append("---")
    
    # Build content
    content_parts = [
        "\n".join(frontmatter_lines),
        "",
        f"# {summary}",
        "",
    ]
    
    if description:
        content_parts.append(description)
    else:
        content_parts.append("*No description provided.*")
    
    content = "\n".join(content_parts)
    
    # Generate filename
    filename = f"{key}-{sanitize_filename(summary)}.md"
    
    return filename, content


def main():
    # Paths
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    input_file = scripts_dir / "postmortems.json"  # JSON stays in scripts/
    output_dir = project_root / "docs"             # Output to project root
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JIRA data
    print(f"Loading JIRA data from {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)
    
    issues = data.get("issues", [])
    print(f"Found {len(issues)} issues")
    
    # Convert and save each issue
    stats = {
        "total": len(issues),
        "success": 0,
        "errors": 0,
        "by_severity": {},
        "by_category": {},
    }
    
    for issue in issues:
        try:
            filename, content = convert_issue_to_markdown(issue)
            output_path = output_dir / filename
            
            with open(output_path, "w") as f:
                f.write(content)
            
            # Track stats
            stats["success"] += 1
            
            # Extract severity for stats
            severity = extract_severity(issue.get("fields", {}).get("description", ""))
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            # Extract category for stats
            category = extract_root_cause_category(issue.get("fields", {}).get("description", ""))
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            print(f"  ✓ {filename}")
            
        except Exception as e:
            stats["errors"] += 1
            print(f"  ✗ Error processing {issue.get('key', 'UNKNOWN')}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total issues:     {stats['total']}")
    print(f"Successfully converted: {stats['success']}")
    print(f"Errors:           {stats['errors']}")
    print(f"\nOutput directory: {output_dir}")
    
    print("\nBy Severity:")
    for sev, count in sorted(stats["by_severity"].items()):
        print(f"  {sev}: {count}")
    
    print("\nBy Root Cause Category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
