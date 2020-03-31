# Cumulative link models for deep ordinal classification

## Algorithms included
This repo contains the code to run experiments with Deep Learning using the Cumulative Link Models and the Quadratic Weighted Kappa loss for the Diabetic Retinopathy, Adience and FGNet datasets. The CLM implementation includes three different link functions used in this work and listed below:

* Logit
* Probit
* Complementary log-log

## Installation

### Dependencies

This repo basically requires:

 * Python         (>= 3.6.8)
 * click          (>=6.7)
 * h5py           (>=2.9.0)
 * Keras          (==2.2.4)
 * matplotlib     (>=3.1.1)
 * numpy          (>=1.17.2)
 * opencv-python  (>=4.1.2)
 * pandas         (>=0.23.4)
 * Pillow         (>=5.2.0)
 * prettytable    (>=0.7.2)
 * scikit-image   (>=0.15.0)
 * scikit-learn   (>=0.21.3)
 * tensorflow     (==1.13.1)

### Compilation

To install the requirements, use:

**Install for CPU**
  `pip install -r requirements.txt`

**Install for GPU**
  `pip install -r requirements_gpu.txt`

## Development

Contributions are welcome. Pull requests are encouraged to be formatted according to [PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using [yapf](https://github.com/google/yapf).

## Usage

You can run all the experiments by running:

```bash
python main_experiment.py experiment -f ../exp/all_experiments.json
```

Note that the Retinopathy dataset must be stored under `../datasets/retinopathy/data128` and the Adience dataset under `../datasets/adience/data256`. This path can be changed by settings the enviroment variable DATASETS_DIR in the execution line:

```bash
DATASETS_DIR=whatever python main_experiment.py experiment -f ../exp/all_experiments.json
```

The `.json` files contain all the details about the experiments settings.

After running the experiments, you can use `tools.py` to watch the results:

```bash
python tools.py
```

## Citation

The paper titled "Cumulative link models for deep ordinal classification" has been published in Neurocomputing. If you use this code, please cite the following paper:

```bibtex
@article{vargas2020cumulative,
  title={Cumulative link models for deep ordinal classification},
  author={Vargas, V{\'\i}ctor Manuel and Guti{\'e}rrez, Pedro Antonio and Herv{\'a}s-Mart{\'\i}nez, C{\'e}sar},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.neucom.2020.03.034}
}
```

## Contributors

#### Cumulative link models for deep ordinal classification

* Víctor Manuel Vargas ([@victormvy](https://github.com/victormvy))
* Pedro Antonio Gutiérrez ([@pagutierrez](https://github.com/pagutierrez))
* César Hervás-Martínez (chervas@uco.es)
