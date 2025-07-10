# Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions

## Installation
python 3.9.0

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 

python main.py --ctrl-type adap_cvarbf --htype dist_cone  

##  arguments:

--ctrl-type: Controller type (cbf, cvarbf, adap_cvarbf)

--htype: Type of h-function (dist_cone, vel, dist)

--beta: risk level, fixed for cvarbf controller and adaptive for adap_cvarbf controller

--S: Number of uncertainty samples

![Overview of Adaptive CVaR Barrier Functions](/config/video20obs/figures/adap_cvarbf_beta0.99_hdist_cone.gif)



---

## Citation

```bibtex
@article{wang2025safe,
  title={Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions},
  author={Wang, Xinyi and Kim, Taekyung and Hoxha, Bardh and Fainekos, Georgios and Panagou, Dimitra},
  conference={IROS},
  year={2025}
}

