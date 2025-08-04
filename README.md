# Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions
https://www.youtube.com/watch?v=VHRnmXToLN8 
## Installation
python 3.9.0

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 

python main.py --ctrl-type adap_cvarbf --htype dist_cone  

## Arguments:

--ctrl-type: Controller type (cbf, cvarbf, adap_cvarbf)

--htype: h-function type (dist_cone, vel, dist)

--beta: Risk parameter: fixed for cvarbf controller and adaptive for adap_cvarbf controller

## Overview of Adaptive CVaR Barrier Functions
![Overview of Adaptive CVaR Barrier Functions](/config/20obs/figures/adap_cvarbf_beta0.99_hdist_cone.gif)



---

## Citation

```bibtex
@article{wang2025safe,
  title={Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions},
  author={Wang, Xinyi and Kim, Taekyung and Hoxha, Bardh and Fainekos, Georgios and Panagou, Dimitra},
  conference={IROS},
  year={2025}
}

