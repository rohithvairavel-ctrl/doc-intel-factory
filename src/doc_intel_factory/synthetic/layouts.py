"""Layout variants — text-based templates mimicking invoices, contracts, tickets."""

from __future__ import annotations

from typing import Any


def render_layout(doc_type: str, fields: dict[str, Any], variant: str, rng) -> str:
    """Render a document as plain text with layout variation."""
    if doc_type == "invoice":
        return _invoice(fields, variant, rng)
    if doc_type == "contract":
        return _contract(fields, variant, rng)
    if doc_type == "ticket":
        return _ticket(fields, variant, rng)
    raise ValueError(f"unknown doc_type={doc_type}")


def _invoice(f: dict[str, Any], variant: str, rng) -> str:
    inv = f["invoice_id"]
    vendor = f["vendor"]
    amount = f["amount"]
    date = f["date"]
    currency = f.get("currency", "USD")
    po = f.get("po_number", "")
    tax = f.get("tax", round(float(amount) * 0.08, 2))

    if variant == "compact":
        return (
            f"INVOICE {inv}\n"
            f"Vendor: {vendor} | Date: {date}\n"
            f"PO: {po}\n"
            f"Total due: {currency} {amount} (tax {tax})\n"
            f"Thank you for your business.\n"
        )
    if variant == "table":
        return (
            f"=== TAX INVOICE ===\n"
            f"Invoice No. .... {inv}\n"
            f"Issue Date ..... {date}\n"
            f"Bill From ...... {vendor}\n"
            f"Purchase Order . {po}\n"
            f"---------------------------\n"
            f"Subtotal ....... {amount}\n"
            f"Tax ............ {tax}\n"
            f"AMOUNT DUE ..... {currency} {amount}\n"
        )
    # narrative / letter style
    return (
        f"{vendor}\nAccounts Receivable\n\n"
        f"Dear Customer,\n"
        f"Please find invoice {inv} dated {date} for services rendered.\n"
        f"Purchase order reference: {po}.\n"
        f"The amount payable is {currency} {amount}, inclusive of tax {tax}.\n"
        f"Remit payment within 30 days.\n"
        f"Sincerely,\n{vendor} Billing\n"
    )


def _contract(f: dict[str, Any], variant: str, rng) -> str:
    parties = f["parties"]
    effective = f["effective_date"]
    term = f["term_months"]
    value = f["contract_value"]
    jurisdiction = f.get("jurisdiction", "Delaware")
    cid = f["contract_id"]

    if variant == "compact":
        return (
            f"CONTRACT {cid}\n"
            f"Parties: {parties}\n"
            f"Effective: {effective} | Term: {term} months\n"
            f"Value: USD {value} | Law: {jurisdiction}\n"
        )
    if variant == "table":
        return (
            f"MASTER SERVICES AGREEMENT\n"
            f"Contract ID ........ {cid}\n"
            f"Counterparty ....... {parties}\n"
            f"Effective Date ..... {effective}\n"
            f"Initial Term ....... {term} months\n"
            f"Contract Value ..... {value} USD\n"
            f"Governing Law ...... {jurisdiction}\n"
        )
    return (
        f"Agreement {cid}\n\n"
        f"This Agreement is entered into as of {effective} by and between {parties}.\n"
        f"The initial term shall be {term} months. The aggregate contract value is "
        f"USD {value}. This Agreement shall be governed by the laws of {jurisdiction}.\n"
    )


def _ticket(f: dict[str, Any], variant: str, rng) -> str:
    tid = f["ticket_id"]
    priority = f["priority"]
    category = f["category"]
    assignee = f["assignee"]
    status = f.get("status", "open")
    summary = f["summary"]

    if variant == "compact":
        return (
            f"[{priority.upper()}] {tid} — {status}\n"
            f"Cat: {category} | Owner: {assignee}\n"
            f"{summary}\n"
        )
    if variant == "table":
        return (
            f"SUPPORT TICKET\n"
            f"ID ......... {tid}\n"
            f"Priority ... {priority}\n"
            f"Category ... {category}\n"
            f"Assignee ... {assignee}\n"
            f"Status ..... {status}\n"
            f"Summary .... {summary}\n"
        )
    return (
        f"Ticket {tid} ({status})\n"
        f"Assigned to {assignee}. Priority marked {priority} under category {category}.\n"
        f"Description: {summary}\n"
    )
