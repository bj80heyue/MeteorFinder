# MeteorFinder
Gifts for landscape photographers. Help the photographer seeking for meteors in the photo sequence.

## Installation

### Requirements
- Python 3.6+
- Numpy 1.15+
- Numba 0.46+
- Opencv 3.0+

### Easy Install
```shell
pip install -r requirements.txt
```

## Getting Started

### Prepare Data
- Put the same batch of photos into a folder.('XXXX/XXX')
	- same batch: Pictures taken in succession from the same camera position.

### Run!
```shell
python findMeteor.py 'XXXX/XXX' [sensitivity_threshold] [rescale]
```
- sensitivity_threshold[10-300][default:30]: The higher the sensitivity threshold, the lower the number of recalls.
- rescale[0.25-1.0][default:0.5]: Affects speed and accuracy.
### Visualization of results


