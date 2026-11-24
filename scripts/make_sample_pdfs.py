"""Generate the sample PDFs in data/sample_pdfs/ used by eval.py.

The three documents are fictional, written for this repo, so the evaluation set can have
exact, checkable answers. Regenerate with:

    uv run python scripts/make_sample_pdfs.py
"""

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_pdfs"

DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "solar_kettle_k2_manual.pdf": (
        "Solar Kettle K2 - User Manual (fictional product)",
        [
            # page 1
            """Safety instructions

Read this manual before using the Solar Kettle K2. The kettle is designed for outdoor use \
with fresh drinking water only. Never fill it with milk, soup or any other liquid, because \
residue can block the steam vent.

Always place the kettle on a flat, stable surface. Do not touch the metal body while the \
water is heating; the handle is the only part that stays cool. Children under 12 should only \
use the kettle under adult supervision.

The maximum fill level is 1.2 litres, marked MAX on the inside of the kettle. The minimum \
fill level is 0.3 litres, marked MIN. Filling above the MAX line can cause boiling water to \
spill from the spout.""",
            # page 2
            """Setup and charging

Unfold the solar panel and point it towards the sun. In full sun the panel charges the \
built-in 12,000 mAh battery from empty to full in about 4 hours. On a cloudy day charging \
can take up to 10 hours. You can also charge the battery from a USB-C wall adapter rated at \
least 30 W, which takes about 2 hours.

A full battery boils 1 litre of water in about 9 minutes, and provides enough energy for \
roughly 5 boils. The battery indicator has four green lights; each light represents about \
25 percent of charge.

To start heating, press and hold the power button for 2 seconds until the ring light turns \
white. The kettle switches itself off automatically when the water boils, or after 15 \
minutes, whichever comes first.""",
            # page 3
            """Troubleshooting

A blinking red light means the kettle is overheating. Switch it off, open the lid and let it \
cool down for 15 minutes before trying again.

Error code E3 means the water level is below the MIN line. Add water and restart the kettle. \
Error code E5 means the battery temperature is too low; the kettle will not heat water when \
the battery is below 0 degrees Celsius. Warm the kettle indoors before use.

If the ring light does not turn on at all, connect the kettle to a USB-C adapter for 10 \
minutes; a fully drained battery needs a short wake-up charge.""",
            # page 4
            """Care and warranty

Descale the kettle once a month with a solution of citric acid and water. Rinse it three \
times with fresh water afterwards. Do not use vinegar, because it can damage the seal of \
the lid.

The Solar Kettle K2 has a 2-year limited warranty from the date of purchase. To activate the \
warranty, register the kettle on the manufacturer's website within 30 days of purchase. The \
warranty does not cover damage caused by dropping the kettle or by using liquids other than \
water.

The solar panel itself is covered for 5 years against loss of output greater than 20 \
percent.""",
        ],
    ),
    "harbor_lane_library_handbook.pdf": (
        "Harbor Lane Community Library - Member Handbook (fictional)",
        [
            # page 1
            """Membership

Membership of the Harbor Lane Community Library is free for anyone who lives, works or \
studies in the district. To join, bring a photo ID and a proof of address issued in the last \
three months. Children under 14 need a parent or guardian to sign the application form.

Your library card is personal and must not be lent to others. If you lose your card, a \
replacement costs 3 dollars. Report a lost card immediately so that nobody else can borrow \
items on your account.""",
            # page 2
            """Borrowing and renewals

Adult members may borrow up to 15 items at a time; junior members may borrow up to 8. Books \
can be borrowed for 21 days. DVDs and video games can be borrowed for 7 days. Magazines can \
be borrowed for 14 days, except the current issue, which must stay in the library.

You can renew an item up to 2 times, online or at the front desk, as long as no other member \
has reserved it. A renewal extends the loan by the original loan period.

Reservations cost nothing. When a reserved item is ready we hold it for you for 5 days at \
the front desk.""",
            # page 3
            """Late returns and fees

There are no late fees for children's materials. For adult members, the late fee is 25 cents \
per item per day, capped at 5 dollars per item. Members with more than 10 dollars in unpaid \
fees cannot borrow new items until the balance is paid.

If an item is 60 days overdue it is treated as lost, and you will be charged its \
replacement cost. If you return a lost item within 6 months, the replacement charge is \
refunded, but late fees still apply.""",
            # page 4
            """Study rooms and opening hours

The library has four quiet study rooms. Members can book a study room for up to 2 hours per \
day. Bookings open 48 hours in advance and can be made online or at the front desk. A room \
is released to the next person if you are more than 15 minutes late.

Opening hours: Monday to Friday from 9:00 to 20:00, Saturday from 10:00 to 17:00. The \
library is closed on Sundays and public holidays. Free Wi-Fi is available throughout the \
building; the network name is HarborLane-Guest.""",
        ],
    ),
    "larkspur_remote_work_policy.pdf": (
        "Larkspur Analytics - Remote Work Policy (fictional company)",
        [
            # page 1
            """Eligibility

This policy applies to all full-time employees of Larkspur Analytics. Employees become \
eligible for remote work after completing their 90-day probation period. Remote work is \
agreed between the employee and their manager and is reviewed every six months.

Employees may work remotely for up to 3 days per week. Teams may set one shared office day \
per week, which everyone on the team is expected to attend in person.""",
            # page 2
            """Working hours and communication

Core hours are 11:00 to 15:00 in the employee's local time zone. During core hours, \
employees must be reachable on the company chat tool and should reply to messages within \
1 hour.

Status must be set to Away when an employee is not at their desk for more than 30 minutes. \
Video should be switched on for team meetings, but may be off for meetings with more than \
20 participants.""",
            # page 3
            """Equipment and expenses

Each remote employee receives a one-time home office stipend of 600 US dollars, renewable \
every 3 years. The stipend can be used for a desk, a chair, a monitor or headphones. \
Company laptops are provided by IT and remain company property.

Internet costs are reimbursed up to 40 US dollars per month. Expense claims must be \
submitted within 30 days of the purchase, with the receipt attached. Claims submitted later \
than 30 days are not reimbursed.""",
            # page 4
            """Security

Employees must use the company VPN whenever they work on a network they do not control, such \
as a cafe, hotel or airport Wi-Fi. Screens must lock automatically after 5 minutes of \
inactivity.

Company data must never be stored on personal devices or personal cloud accounts. A lost or \
stolen laptop, or any suspected security incident, must be reported to the security desk \
within 1 hour of discovery. Printing confidential documents at home is not permitted.""",
        ],
    ),
}


def build_pdf(title: str, pages: list[str]) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=False)
    for number, body in enumerate(pages, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 10, f"Page {number}", align="C")
    return bytes(pdf.output())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, pages) in DOCUMENTS.items():
        (OUT_DIR / filename).write_bytes(build_pdf(title, pages))
        print(f"wrote {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
