# Forensic Wound Classification

Classifying forensic wound photographs into 7 wound types, and testing whether cropping to the wound region helps.

## Wound types

Defined after DiMaio's Forensic Pathology (DiMaio, 2021), as in the thesis proposal. A photo with mixed features takes the dominant type documented in the case record and confirmed by expert review.

**Cut wound**:
An incised sharp-force injury, generally longer than it is deep, with clean-cut margins.
_Avoid_: incision (as a class name), slash

**Graze wound**:
A superficial scrape-type abrasion made by tangential friction against the skin.
_Avoid_: abrasion (as a class name)

**Scratch**:
A narrow, linear scrape abrasion.
_Avoid_: graze (for a linear mark)

**Contusion**:
Haemorrhage into soft tissue from blood vessels ruptured by blunt trauma.
_Avoid_: bruise (as a class name)

**Laceration**:
A tear of skin or soft tissue from shearing or crushing blunt force; typically irregular, with abraded or contused margins and possible tissue bridging.
_Avoid_: tear, cut (for blunt-force wounds)

**Penetration wound**:
A non-firearm wound made by a pointed object entering the skin or deeper tissue, including stab and puncture injuries when the case record supports it.
_Avoid_: penetrating wound (sharp & blunt), stab wound (as a class name)

**Gunshot wound**:
A ballistic wound from a firearm projectile, including firearm graze or tangential wounds when the case record identifies them as firearm-related.
_Avoid_: GSW (in prose)

## Wound regions

**Manual box**:
A rectangle around one wound, drawn by a person in CVAT; its class is the photo's wound type.
_Avoid_: annotation, ground-truth box (when the source matters)

**Auto box**:
A rectangle around one wound, produced by a wound detector rather than a person.
_Avoid_: prediction, detection, pseudo-label

**Wound detector**:
A model that outputs auto boxes for a photo, with no wound type.
_Avoid_: annotator, localiser

**Detector arm**:
One of the three sources of auto boxes in Experiment 4: **4Z** zero-shot (no training), **4P** public wound model, **4F** fine-tuned on manual boxes of the training split.
_Avoid_: detector variant, setting

## ROI designs

**Union ROI**:
One crop per photo, covering all of its boxes (Experiment 1).
_Avoid_: ROI (alone)

**Per-box ROI**:
One crop per box, the photo predicted from the mean of its crops (Experiment 2).
_Avoid_: box ROI

**Per-crop ROI**:
One crop per box, each crop a sample of its own, scored per crop (Experiment 3).
_Avoid_: crop ROI (alone)

## Experiments

**Experiment 4 (auto-box ROI)**:
Experiments 1–3 repeated with auto boxes in place of manual boxes, for each detector arm.
_Avoid_: auto-annotation experiment
