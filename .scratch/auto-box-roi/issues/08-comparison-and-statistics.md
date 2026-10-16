# Classification comparisons and multiplicity

Type: grilling
Status: open
Blocked by: none

## Question

Which comparisons are primary and secondary, and over which set is the Holm correction applied? Candidates:
- **Primary**: auto-ROI (existing models) − manual-ROI, paired per family. Run per detector arm, for union and per-box designs: 8 families × 3 arms × 2 designs.
- **Secondary**:
  - retrained auto-ROI − manual-ROI;
  - auto-ROI − full image.

Also decide:
- whether the arms are compared with each other;
- whether results are pooled over families (as in the README's main table);
- the bootstrap settings (2,000 case resamples, seed 42, as before?).
