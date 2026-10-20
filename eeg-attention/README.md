Python/MNE analysis pipeline for an EEG study of auditory selective attention in normal-hearing and hearing-impaired listeners

## Background
Clinical care for sensorineural hearing loss (SNHL) focuses on audibility,
yet many patients still struggle to understand speech in noise even with
amplification. While most SNHL research targets sensory deficits, little is
known about how top-down selective attention is affected. This project uses
simultaneous EEG and behavioral measures to test whether attentional control
differs in listeners with hearing loss.

## Experimental design
- **Recording:** 32-channel EEG with simultaneous behavioral responses
- **Task:** speech-on-speech selective attention built from matrix sentence
  materials; target and masker use the same talker and word set at 0 dB
  target-to-masker ratio
- **Spatial cues:** streams lateralized to opposite hemifields using
  interaural time differences and staggered in time, minimizing acoustic
  overlap so measures reflect top-down attention rather than peripheral
  deficits
- Target location and stimuli counterbalanced across trials
  
## Analysis methods
- **Alpha lateralization and Attentional modulation of evoked response** (`Individual_Final.py`) : multitaper estimates
  of alpha-band power (7–15 Hz) over parieto-occipital channels during the
  preparatory period, compared between attend-left and attend-right trials
- **Group analysis** (`HL_GA.py`): grand averages compared across
  normal-hearing and hearing-loss groups
