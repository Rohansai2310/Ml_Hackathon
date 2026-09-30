# Blocking V1: inspected validation misses

The full six-route run missed 142,440 of 764,074 validation true links. The
saved blocking_misses_v1.tsv contains the first 1,000 missed links in sorted
S1 order; blocking_miss_ids_v1.tsv.gz contains every missed ID pair. The rows
below were selected from that bounded inspection sample to show distinct
failure modes. They are examples, not estimates of category prevalence.

| S1 ID → target ID | S1 name | True target name | Visible reason V1 misses it |
|---|---|---|---|
| S1-100011573 → S2-772485930 | Seven Energy Private Limited | सेवन एनर्जी प्राइवेट लिमिटेड | Latin/Devanagari name transliteration; the two-digit address number 42 does not meet the strong-address rule. |
| S1-100124296 → S2-684749555 | Raj Finance Private Limited | রাজ ফাইন্যান্স প্রাইভেট লিমিটেড | Latin/Bengali script change. |
| S1-100034634 → S3-181606392 | One Systems LLP | वन सिस्टम्स एलएलपी | Script change plus address number 1689 versus 01689. |
| S1-100074352 → S3-198136359 | Heart Center | Heart Cénter | Accent changes a name token and the target omits the house number. |
| S1-10010336 → S3-278428302 | Nagpur Wellness Pvt Ltd | Nagpur Wlelnregs Pvt Ltd | Name corruption and shorter target address. |
| S1-100122317 → S2-672079026 | New Delhi Apps Private Limited | newdelhiapps.com | Website form concatenates the name words; address numbers also differ in formatting. |
| S1-100207029 → S3-840393579 | G S & F Berto-Eufaula | G S & F Berto-Eufau1a | A letter/digit typo and padded address number 213 versus 00213. |
| S1-100333402 → S2-72321242 | Innovative Intelligence Services LLC | Innovative Services LLC Services | Name tokens are dropped/reordered and the target address is blank. |

The country labels agree in these inspected examples. Full validation recall
is 81.36% overall, 80.36% for S2, and 82.30% for S3. Fuzzy and transliteration
retrieval in a later phase is expected to recover some of these missed links.
