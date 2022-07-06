## PyCaret
- 개발환경 설치
- https://pycaret.gitbook.io/docs/
- 로컬에서 하기 힘듬
- git hub > requirements.txt > 기본환경 버전 확인 및 설치
- https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html

- PyCaret 설치


```python
!pip uninstall sklearn -y
!pip install --upgrade sklearn
!pip install scikit-learn==0.23.2 --user
!pip install pycaret
!pip install markupsafe==2.0.1
```

    Found existing installation: sklearn 0.0
    Uninstalling sklearn-0.0:
      Successfully uninstalled sklearn-0.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting sklearn
      Downloading sklearn-0.0.tar.gz (1.1 kB)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from sklearn) (1.0.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn) (3.1.0)
    Requirement already satisfied: numpy>=1.14.6 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn) (1.21.6)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn) (1.1.0)
    Building wheels for collected packages: sklearn
      Building wheel for sklearn (setup.py) ... [?25l[?25hdone
      Created wheel for sklearn: filename=sklearn-0.0-py2.py3-none-any.whl size=1310 sha256=9e2009e09e4eff4f12aba4171158500c2dba9a70171e14e5dc159fc591c858b8
      Stored in directory: /root/.cache/pip/wheels/46/ef/c3/157e41f5ee1372d1be90b09f74f82b10e391eaacca8f22d33e
    Successfully built sklearn
    Installing collected packages: sklearn
    Successfully installed sklearn-0.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting scikit-learn==0.23.2
      Downloading scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl (6.8 MB)
    [K     |████████████████████████████████| 6.8 MB 5.1 MB/s 
    [?25hRequirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2) (1.21.6)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2) (1.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2) (1.4.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2) (3.1.0)
    Installing collected packages: scikit-learn
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    yellowbrick 1.4 requires scikit-learn>=1.0.0, but you have scikit-learn 0.23.2 which is incompatible.
    imbalanced-learn 0.8.1 requires scikit-learn>=0.24, but you have scikit-learn 0.23.2 which is incompatible.[0m
    Successfully installed scikit-learn-0.23.2
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pycaret
      Downloading pycaret-2.3.10-py3-none-any.whl (320 kB)
    [K     |████████████████████████████████| 320 kB 5.1 MB/s 
    [?25hRequirement already satisfied: scikit-learn==0.23.2 in /root/.local/lib/python3.7/site-packages (from pycaret) (0.23.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from pycaret) (3.2.2)
    Requirement already satisfied: textblob in /usr/local/lib/python3.7/dist-packages (from pycaret) (0.15.3)
    Requirement already satisfied: gensim<4.0.0 in /usr/local/lib/python3.7/dist-packages (from pycaret) (3.6.0)
    Collecting pyod
      Downloading pyod-1.0.2.tar.gz (122 kB)
    [K     |████████████████████████████████| 122 kB 63.0 MB/s 
    [?25hRequirement already satisfied: pyyaml<6.0.0 in /usr/local/lib/python3.7/dist-packages (from pycaret) (3.13)
    Collecting kmodes>=0.10.1
      Downloading kmodes-0.12.1-py2.py3-none-any.whl (20 kB)
    Requirement already satisfied: plotly>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from pycaret) (5.5.0)
    Collecting mlxtend>=0.17.0
      Downloading mlxtend-0.20.0-py2.py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 41.6 MB/s 
    [?25hCollecting scikit-plot
      Downloading scikit_plot-0.3.7-py3-none-any.whl (33 kB)
    Requirement already satisfied: wordcloud in /usr/local/lib/python3.7/dist-packages (from pycaret) (1.5.0)
    Collecting pyLDAvis
      Downloading pyLDAvis-3.3.1.tar.gz (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 40.9 MB/s 
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from pycaret) (1.1.0)
    Collecting Boruta
      Downloading Boruta-0.3-py3-none-any.whl (56 kB)
    [K     |████████████████████████████████| 56 kB 4.0 MB/s 
    [?25hRequirement already satisfied: scipy<=1.5.4 in /usr/local/lib/python3.7/dist-packages (from pycaret) (1.4.1)
    Requirement already satisfied: IPython in /usr/local/lib/python3.7/dist-packages (from pycaret) (5.5.0)
    Collecting pandas-profiling>=2.8.0
      Downloading pandas_profiling-3.2.0-py2.py3-none-any.whl (262 kB)
    [K     |████████████████████████████████| 262 kB 51.5 MB/s 
    [?25hCollecting lightgbm>=2.3.1
      Downloading lightgbm-3.3.2-py3-none-manylinux1_x86_64.whl (2.0 MB)
    [K     |████████████████████████████████| 2.0 MB 38.0 MB/s 
    [?25hRequirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from pycaret) (0.11.2)
    Requirement already satisfied: ipywidgets in /usr/local/lib/python3.7/dist-packages (from pycaret) (7.7.0)
    Collecting umap-learn
      Downloading umap-learn-0.5.3.tar.gz (88 kB)
    [K     |████████████████████████████████| 88 kB 6.8 MB/s 
    [?25hRequirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (from pycaret) (3.7)
    Requirement already satisfied: yellowbrick>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from pycaret) (1.4)
    Collecting mlflow
      Downloading mlflow-1.27.0-py3-none-any.whl (17.9 MB)
    [K     |████████████████████████████████| 17.9 MB 41.3 MB/s 
    [?25hCollecting spacy<2.4.0
      Downloading spacy-2.3.7-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.4 MB)
    [K     |████████████████████████████████| 10.4 MB 24.8 MB/s 
    [?25hCollecting imbalanced-learn==0.7.0
      Downloading imbalanced_learn-0.7.0-py3-none-any.whl (167 kB)
    [K     |████████████████████████████████| 167 kB 53.7 MB/s 
    [?25hRequirement already satisfied: cufflinks>=0.17.0 in /usr/local/lib/python3.7/dist-packages (from pycaret) (0.17.3)
    Requirement already satisfied: numba<0.55 in /usr/local/lib/python3.7/dist-packages (from pycaret) (0.51.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from pycaret) (1.3.5)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn==0.7.0->pycaret) (1.21.6)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2->pycaret) (3.1.0)
    Requirement already satisfied: colorlover>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from cufflinks>=0.17.0->pycaret) (0.3.0)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from cufflinks>=0.17.0->pycaret) (1.15.0)
    Requirement already satisfied: setuptools>=34.4.1 in /usr/local/lib/python3.7/dist-packages (from cufflinks>=0.17.0->pycaret) (57.4.0)
    Requirement already satisfied: smart-open>=1.2.1 in /usr/local/lib/python3.7/dist-packages (from gensim<4.0.0->pycaret) (5.2.1)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (1.0.18)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (5.1.1)
    Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (4.8.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (4.4.2)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (0.7.5)
    Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (2.6.1)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from IPython->pycaret) (0.8.1)
    Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->pycaret) (4.10.1)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->pycaret) (5.4.0)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->pycaret) (0.2.0)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->pycaret) (1.1.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets->pycaret) (3.6.0)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets->pycaret) (5.3.5)
    Requirement already satisfied: tornado>=4.0 in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets->pycaret) (5.1.1)
    Requirement already satisfied: wheel in /usr/local/lib/python3.7/dist-packages (from lightgbm>=2.3.1->pycaret) (0.37.1)
    Collecting mlxtend>=0.17.0
      Downloading mlxtend-0.19.0-py2.py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 54.0 MB/s 
    [?25hRequirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pycaret) (1.4.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pycaret) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pycaret) (2.8.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->pycaret) (0.11.0)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->pycaret) (4.1.1)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets->pycaret) (4.3.3)
    Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets->pycaret) (2.15.3)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets->pycaret) (4.10.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets->pycaret) (4.11.4)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets->pycaret) (21.4.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets->pycaret) (0.18.1)
    Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets->pycaret) (5.7.1)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat>=4.2.0->ipywidgets->pycaret) (3.8.0)
    Requirement already satisfied: llvmlite<0.35,>=0.34.0.dev0 in /usr/local/lib/python3.7/dist-packages (from numba<0.55->pycaret) (0.34.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->pycaret) (2022.1)
    Requirement already satisfied: tqdm>=4.48.2 in /usr/local/lib/python3.7/dist-packages (from pandas-profiling>=2.8.0->pycaret) (4.64.0)
    Collecting tangled-up-in-unicode==0.2.0
      Downloading tangled_up_in_unicode-0.2.0-py3-none-any.whl (4.7 MB)
    [K     |████████████████████████████████| 4.7 MB 43.0 MB/s 
    [?25hCollecting pyyaml<6.0.0
      Downloading PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)
    [K     |████████████████████████████████| 636 kB 57.9 MB/s 
    [?25hCollecting htmlmin>=0.1.12
      Downloading htmlmin-0.1.12.tar.gz (19 kB)
    Requirement already satisfied: jinja2>=2.11.1 in /usr/local/lib/python3.7/dist-packages (from pandas-profiling>=2.8.0->pycaret) (2.11.3)
    Collecting multimethod>=1.4
      Downloading multimethod-1.8-py3-none-any.whl (9.8 kB)
    Requirement already satisfied: pydantic>=1.8.1 in /usr/local/lib/python3.7/dist-packages (from pandas-profiling>=2.8.0->pycaret) (1.8.2)
    Requirement already satisfied: missingno>=0.4.2 in /usr/local/lib/python3.7/dist-packages (from pandas-profiling>=2.8.0->pycaret) (0.5.1)
    Collecting requests>=2.24.0
      Downloading requests-2.28.1-py3-none-any.whl (62 kB)
    [K     |████████████████████████████████| 62 kB 1.5 MB/s 
    [?25hCollecting visions[type_image_path]==0.7.4
      Downloading visions-0.7.4-py3-none-any.whl (102 kB)
    [K     |████████████████████████████████| 102 kB 11.5 MB/s 
    [?25hCollecting phik>=0.11.1
      Downloading phik-0.12.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (690 kB)
    [K     |████████████████████████████████| 690 kB 58.8 MB/s 
    [?25hCollecting markupsafe~=2.1.1
      Downloading MarkupSafe-2.1.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Requirement already satisfied: networkx>=2.4 in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.7.4->pandas-profiling>=2.8.0->pycaret) (2.6.3)
    Collecting imagehash
      Downloading ImageHash-4.2.1.tar.gz (812 kB)
    [K     |████████████████████████████████| 812 kB 56.6 MB/s 
    [?25hRequirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.7.4->pandas-profiling>=2.8.0->pycaret) (7.1.2)
    Collecting scipy<=1.5.4
      Downloading scipy-1.5.4-cp37-cp37m-manylinux1_x86_64.whl (25.9 MB)
    [K     |████████████████████████████████| 25.9 MB 1.2 MB/s 
    [?25hRequirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly>=4.4.1->pycaret) (8.0.1)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->IPython->pycaret) (0.2.5)
    Requirement already satisfied: charset-normalizer<3,>=2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2022.6.15)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.24.0->pandas-profiling>=2.8.0->pycaret) (1.24.3)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<2.4.0->pycaret) (2.0.6)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy<2.4.0->pycaret) (0.7.7)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy<2.4.0->pycaret) (1.0.7)
    Collecting thinc<7.5.0,>=7.4.1
      Downloading thinc-7.4.5-cp37-cp37m-manylinux2014_x86_64.whl (1.0 MB)
    [K     |████████████████████████████████| 1.0 MB 54.7 MB/s 
    [?25hCollecting catalogue<1.1.0,>=0.0.7
      Downloading catalogue-1.0.0-py2.py3-none-any.whl (7.7 kB)
    Collecting plac<1.2.0,>=0.9.6
      Downloading plac-1.1.3-py2.py3-none-any.whl (20 kB)
    Collecting srsly<1.1.0,>=1.0.2
      Downloading srsly-1.0.5-cp37-cp37m-manylinux2014_x86_64.whl (184 kB)
    [K     |████████████████████████████████| 184 kB 51.4 MB/s 
    [?25hRequirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy<2.4.0->pycaret) (0.9.1)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<2.4.0->pycaret) (3.0.6)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.3.1)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.13.3)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (1.8.0)
    Requirement already satisfied: nbconvert in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.6.1)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.7/dist-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets->pycaret) (23.1.0)
    Requirement already satisfied: ptyprocess in /usr/local/lib/python3.7/dist-packages (from terminado>=0.8.1->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.7.0)
    Collecting yellowbrick>=1.0.1
      Downloading yellowbrick-1.3.post1-py3-none-any.whl (271 kB)
    [K     |████████████████████████████████| 271 kB 56.0 MB/s 
    [?25hCollecting numpy>=1.13.3
      Downloading numpy-1.19.5-cp37-cp37m-manylinux2010_x86_64.whl (14.8 MB)
    [K     |████████████████████████████████| 14.8 MB 42.1 MB/s 
    [?25hRequirement already satisfied: PyWavelets in /usr/local/lib/python3.7/dist-packages (from imagehash->visions[type_image_path]==0.7.4->pandas-profiling>=2.8.0->pycaret) (1.3.0)
    Requirement already satisfied: sqlparse>=0.3.1 in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (0.4.2)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (1.3.0)
    Requirement already satisfied: sqlalchemy>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (1.4.37)
    Collecting querystring-parser
      Downloading querystring_parser-1.2.4-py2.py3-none-any.whl (7.9 kB)
    Collecting gunicorn
      Downloading gunicorn-20.1.0-py3-none-any.whl (79 kB)
    [K     |████████████████████████████████| 79 kB 8.8 MB/s 
    [?25hCollecting databricks-cli>=0.8.7
      Downloading databricks-cli-0.17.0.tar.gz (81 kB)
    [K     |████████████████████████████████| 81 kB 8.8 MB/s 
    [?25hRequirement already satisfied: click>=7.0 in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (7.1.2)
    Collecting gitpython>=2.1.0
      Downloading GitPython-3.1.27-py3-none-any.whl (181 kB)
    [K     |████████████████████████████████| 181 kB 45.0 MB/s 
    [?25hCollecting prometheus-flask-exporter
      Downloading prometheus_flask_exporter-0.20.2-py3-none-any.whl (18 kB)
    Requirement already satisfied: Flask in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (1.1.4)
    Requirement already satisfied: protobuf>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (3.17.3)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (21.3)
    Requirement already satisfied: entrypoints in /usr/local/lib/python3.7/dist-packages (from mlflow->pycaret) (0.4)
    Collecting alembic
      Downloading alembic-1.8.0-py3-none-any.whl (209 kB)
    [K     |████████████████████████████████| 209 kB 58.4 MB/s 
    [?25hCollecting docker>=4.0.0
      Downloading docker-5.0.3-py2.py3-none-any.whl (146 kB)
    [K     |████████████████████████████████| 146 kB 45.5 MB/s 
    [?25hCollecting pyjwt>=1.7.0
      Downloading PyJWT-2.4.0-py3-none-any.whl (18 kB)
    Requirement already satisfied: oauthlib>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from databricks-cli>=0.8.7->mlflow->pycaret) (3.2.0)
    Requirement already satisfied: tabulate>=0.7.7 in /usr/local/lib/python3.7/dist-packages (from databricks-cli>=0.8.7->mlflow->pycaret) (0.8.9)
    Collecting websocket-client>=0.32.0
      Downloading websocket_client-1.3.3-py3-none-any.whl (54 kB)
    [K     |████████████████████████████████| 54 kB 2.4 MB/s 
    [?25hCollecting gitdb<5,>=4.0.1
      Downloading gitdb-4.0.9-py3-none-any.whl (63 kB)
    [K     |████████████████████████████████| 63 kB 1.4 MB/s 
    [?25hCollecting smmap<6,>=3.0.1
      Downloading smmap-5.0.0-py3-none-any.whl (24 kB)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.7/dist-packages (from sqlalchemy>=1.4.0->mlflow->pycaret) (1.1.2)
    Collecting Mako
      Downloading Mako-1.2.1-py3-none-any.whl (78 kB)
    [K     |████████████████████████████████| 78 kB 7.0 MB/s 
    [?25hRequirement already satisfied: itsdangerous<2.0,>=0.24 in /usr/local/lib/python3.7/dist-packages (from Flask->mlflow->pycaret) (1.1.0)
    Requirement already satisfied: Werkzeug<2.0,>=0.15 in /usr/local/lib/python3.7/dist-packages (from Flask->mlflow->pycaret) (1.0.1)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.8.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.0.0)
    Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.6.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (1.5.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.7.1)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.5.1)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.7/dist-packages (from nltk->pycaret) (2022.6.2)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.7/dist-packages (from prometheus-flask-exporter->mlflow->pycaret) (0.14.1)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from pyLDAvis->pycaret) (0.16.0)
    Collecting funcy
      Downloading funcy-1.17-py2.py3-none-any.whl (33 kB)
    Collecting pyLDAvis
      Downloading pyLDAvis-3.3.0.tar.gz (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 33.8 MB/s 
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
      Downloading pyLDAvis-3.2.2.tar.gz (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 51.9 MB/s 
    [?25hRequirement already satisfied: numexpr in /usr/local/lib/python3.7/dist-packages (from pyLDAvis->pycaret) (2.8.1)
    Requirement already satisfied: statsmodels in /usr/local/lib/python3.7/dist-packages (from pyod->pycaret) (0.10.2)
    Requirement already satisfied: patsy>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from statsmodels->pyod->pycaret) (0.5.2)
    Collecting pynndescent>=0.5
      Downloading pynndescent-0.5.7.tar.gz (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 50.0 MB/s 
    [?25hBuilding wheels for collected packages: htmlmin, imagehash, databricks-cli, pyLDAvis, pyod, umap-learn, pynndescent
      Building wheel for htmlmin (setup.py) ... [?25l[?25hdone
      Created wheel for htmlmin: filename=htmlmin-0.1.12-py3-none-any.whl size=27098 sha256=2d331816ce85c813189d9041cd350f13f87473bc4ed0e61a626eefabdd474bea
      Stored in directory: /root/.cache/pip/wheels/70/e1/52/5b14d250ba868768823940c3229e9950d201a26d0bd3ee8655
      Building wheel for imagehash (setup.py) ... [?25l[?25hdone
      Created wheel for imagehash: filename=ImageHash-4.2.1-py2.py3-none-any.whl size=295206 sha256=0a50a223aec3e2fe143a05448371c8f6ba3661c9780100762110cb0a321fef5f
      Stored in directory: /root/.cache/pip/wheels/4c/d5/59/5e3e297533ddb09407769762985d134135064c6831e29a914e
      Building wheel for databricks-cli (setup.py) ... [?25l[?25hdone
      Created wheel for databricks-cli: filename=databricks_cli-0.17.0-py3-none-any.whl size=141960 sha256=a0c3c537ebb9f15c40226c1b1a9d5460c51d9ad7df6fcc62f984cd1477312395
      Stored in directory: /root/.cache/pip/wheels/55/c3/db/33705569425fd2bdc9ea73051a8053fa26965c2bce8a146747
      Building wheel for pyLDAvis (setup.py) ... [?25l[?25hdone
      Created wheel for pyLDAvis: filename=pyLDAvis-3.2.2-py2.py3-none-any.whl size=135617 sha256=b998686d5dc7de493960ef0b985c3aad55f9d1f9648aefa1d83abddcca4b1720
      Stored in directory: /root/.cache/pip/wheels/f8/b1/9b/560ac1931796b7303f7b517b949d2d31a4fbc512aad3b9f284
      Building wheel for pyod (setup.py) ... [?25l[?25hdone
      Created wheel for pyod: filename=pyod-1.0.2-py3-none-any.whl size=150272 sha256=0b118bffd8eb7273753256e07f488717bbd477150bb571c69fd169acef28a781
      Stored in directory: /root/.cache/pip/wheels/e6/8f/06/5512935ed3c79659f612e8bb8f43cb51dd47c21973e0230997
      Building wheel for umap-learn (setup.py) ... [?25l[?25hdone
      Created wheel for umap-learn: filename=umap_learn-0.5.3-py3-none-any.whl size=82829 sha256=a259ebbb4f813fe214b294836aba0dc730e79022e68603c096508bc89d5c639f
      Stored in directory: /root/.cache/pip/wheels/b3/52/a5/1fd9e3e76a7ab34f134c07469cd6f16e27ef3a37aeff1fe821
      Building wheel for pynndescent (setup.py) ... [?25l[?25hdone
      Created wheel for pynndescent: filename=pynndescent-0.5.7-py3-none-any.whl size=54286 sha256=188344a0b7ea6a595ea7e57f1da5f50d68959c7c226ccd57a5ebe4d010061f03
      Stored in directory: /root/.cache/pip/wheels/7f/2a/f8/7bd5dcec71bd5c669f6f574db3113513696b98f3f9b51f496c
    Successfully built htmlmin imagehash databricks-cli pyLDAvis pyod umap-learn pynndescent
    Installing collected packages: markupsafe, numpy, tangled-up-in-unicode, smmap, scipy, multimethod, websocket-client, visions, srsly, requests, pyjwt, plac, Mako, imagehash, gitdb, catalogue, thinc, querystring-parser, pyyaml, pynndescent, prometheus-flask-exporter, phik, htmlmin, gunicorn, gitpython, funcy, docker, databricks-cli, alembic, yellowbrick, umap-learn, spacy, scikit-plot, pyod, pyLDAvis, pandas-profiling, mlxtend, mlflow, lightgbm, kmodes, imbalanced-learn, Boruta, pycaret
      Attempting uninstall: markupsafe
        Found existing installation: MarkupSafe 2.0.1
        Uninstalling MarkupSafe-2.0.1:
          Successfully uninstalled MarkupSafe-2.0.1
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.6
        Uninstalling numpy-1.21.6:
          Successfully uninstalled numpy-1.21.6
      Attempting uninstall: scipy
        Found existing installation: scipy 1.4.1
        Uninstalling scipy-1.4.1:
          Successfully uninstalled scipy-1.4.1
      Attempting uninstall: srsly
        Found existing installation: srsly 2.4.3
        Uninstalling srsly-2.4.3:
          Successfully uninstalled srsly-2.4.3
      Attempting uninstall: requests
        Found existing installation: requests 2.23.0
        Uninstalling requests-2.23.0:
          Successfully uninstalled requests-2.23.0
      Attempting uninstall: catalogue
        Found existing installation: catalogue 2.0.7
        Uninstalling catalogue-2.0.7:
          Successfully uninstalled catalogue-2.0.7
      Attempting uninstall: thinc
        Found existing installation: thinc 8.0.17
        Uninstalling thinc-8.0.17:
          Successfully uninstalled thinc-8.0.17
      Attempting uninstall: pyyaml
        Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
      Attempting uninstall: yellowbrick
        Found existing installation: yellowbrick 1.4
        Uninstalling yellowbrick-1.4:
          Successfully uninstalled yellowbrick-1.4
      Attempting uninstall: spacy
        Found existing installation: spacy 3.3.1
        Uninstalling spacy-3.3.1:
          Successfully uninstalled spacy-3.3.1
      Attempting uninstall: pandas-profiling
        Found existing installation: pandas-profiling 1.4.1
        Uninstalling pandas-profiling-1.4.1:
          Successfully uninstalled pandas-profiling-1.4.1
      Attempting uninstall: mlxtend
        Found existing installation: mlxtend 0.14.0
        Uninstalling mlxtend-0.14.0:
          Successfully uninstalled mlxtend-0.14.0
      Attempting uninstall: lightgbm
        Found existing installation: lightgbm 2.2.3
        Uninstalling lightgbm-2.2.3:
          Successfully uninstalled lightgbm-2.2.3
      Attempting uninstall: imbalanced-learn
        Found existing installation: imbalanced-learn 0.8.1
        Uninstalling imbalanced-learn-0.8.1:
          Successfully uninstalled imbalanced-learn-0.8.1
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    xarray-einstats 0.2.2 requires numpy>=1.21, but you have numpy 1.19.5 which is incompatible.
    tensorflow 2.8.2+zzzcolab20220527125636 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    google-colab 1.0.0 requires requests~=2.23.0, but you have requests 2.28.1 which is incompatible.
    en-core-web-sm 3.3.0 requires spacy<3.4.0,>=3.3.0.dev0, but you have spacy 2.3.7 which is incompatible.
    datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
    albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.[0m
    Successfully installed Boruta-0.3 Mako-1.2.1 alembic-1.8.0 catalogue-1.0.0 databricks-cli-0.17.0 docker-5.0.3 funcy-1.17 gitdb-4.0.9 gitpython-3.1.27 gunicorn-20.1.0 htmlmin-0.1.12 imagehash-4.2.1 imbalanced-learn-0.7.0 kmodes-0.12.1 lightgbm-3.3.2 markupsafe-2.1.1 mlflow-1.27.0 mlxtend-0.19.0 multimethod-1.8 numpy-1.19.5 pandas-profiling-3.2.0 phik-0.12.2 plac-1.1.3 prometheus-flask-exporter-0.20.2 pyLDAvis-3.2.2 pycaret-2.3.10 pyjwt-2.4.0 pynndescent-0.5.7 pyod-1.0.2 pyyaml-5.4.1 querystring-parser-1.2.4 requests-2.28.1 scikit-plot-0.3.7 scipy-1.5.4 smmap-5.0.0 spacy-2.3.7 srsly-1.0.5 tangled-up-in-unicode-0.2.0 thinc-7.4.5 umap-learn-0.5.3 visions-0.7.4 websocket-client-1.3.3 yellowbrick-1.3.post1
    



    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting markupsafe==2.0.1
      Downloading MarkupSafe-2.0.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (31 kB)
    Installing collected packages: markupsafe
      Attempting uninstall: markupsafe
        Found existing installation: MarkupSafe 2.1.1
        Uninstalling MarkupSafe-2.1.1:
          Successfully uninstalled MarkupSafe-2.1.1
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pandas-profiling 3.2.0 requires markupsafe~=2.1.1, but you have markupsafe 2.0.1 which is incompatible.
    datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.[0m
    Successfully installed markupsafe-2.0.1
    

- 설치 완료 후 RESTART RUNTIME 클릭

- PyCaret를 구글 코랩에서 활성화


```python
from pycaret.utils import enable_colab
enable_colab()
```

    Colab mode enabled.
    


```python
from pycaret.datasets import get_data
dataset = get_data('credit')  # 샘플 데이터
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>





  <div id="df-2453a33b-785c-49ea-88e2-103241b65c49">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>19394.0</td>
      <td>19619.0</td>
      <td>20024.0</td>
      <td>2500.0</td>
      <td>1815.0</td>
      <td>657.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>800.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2453a33b-785c-49ea-88e2-103241b65c49')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2453a33b-785c-49ea-88e2-103241b65c49 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2453a33b-785c-49ea-88e2-103241b65c49');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




```python
dataset.info()
# 오! 다! 숫자! PCA!!
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>



    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24000 entries, 0 to 23999
    Data columns (total 24 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   LIMIT_BAL  24000 non-null  int64  
     1   SEX        24000 non-null  int64  
     2   EDUCATION  24000 non-null  int64  
     3   MARRIAGE   24000 non-null  int64  
     4   AGE        24000 non-null  int64  
     5   PAY_1      24000 non-null  int64  
     6   PAY_2      24000 non-null  int64  
     7   PAY_3      24000 non-null  int64  
     8   PAY_4      24000 non-null  int64  
     9   PAY_5      24000 non-null  int64  
     10  PAY_6      24000 non-null  int64  
     11  BILL_AMT1  24000 non-null  float64
     12  BILL_AMT2  24000 non-null  float64
     13  BILL_AMT3  24000 non-null  float64
     14  BILL_AMT4  24000 non-null  float64
     15  BILL_AMT5  24000 non-null  float64
     16  BILL_AMT6  24000 non-null  float64
     17  PAY_AMT1   24000 non-null  float64
     18  PAY_AMT2   24000 non-null  float64
     19  PAY_AMT3   24000 non-null  float64
     20  PAY_AMT4   24000 non-null  float64
     21  PAY_AMT5   24000 non-null  float64
     22  PAY_AMT6   24000 non-null  float64
     23  default    24000 non-null  int64  
    dtypes: float64(12), int64(12)
    memory usage: 4.4 MB
    


```python
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>



    Data for Modeling: (22800, 24)
    Unseen Data For Predictions: (1200, 24)
    

- setup
- 자기가 알아서 다 설정해줌


```python
import jinja2
from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'default', pca=True, pca_components=10, feature_selection=True, session_id=123) 
```



  <div id="df-a2451dab-0bdb-4a90-a7b1-33e23d5f9739">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Description</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>session_id</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Target</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target Type</td>
      <td>Binary</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Label Encoded</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Original Data</td>
      <td>(22800, 24)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Missing Values</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Numeric Features</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Categorical Features</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ordinal Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>High Cardinality Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>High Cardinality Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Transformed Train Set</td>
      <td>(15959, 10)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Transformed Test Set</td>
      <td>(6841, 10)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Shuffle Train-Test</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Stratify Train-Test</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Fold Generator</td>
      <td>StratifiedKFold</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Fold Number</td>
      <td>10</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CPU Jobs</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Use GPU</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Log Experiment</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Experiment Name</td>
      <td>clf-default-name</td>
    </tr>
    <tr>
      <th>21</th>
      <td>USI</td>
      <td>1138</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Imputation Type</td>
      <td>simple</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Iterative Imputation Iteration</td>
      <td>None</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Numeric Imputer</td>
      <td>mean</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Iterative Imputation Numeric Model</td>
      <td>None</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Categorical Imputer</td>
      <td>constant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Iterative Imputation Categorical Model</td>
      <td>None</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Unknown Categoricals Handling</td>
      <td>least_frequent</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Normalize</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Normalize Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Transformation</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Transformation Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PCA</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PCA Method</td>
      <td>linear</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PCA Components</td>
      <td>10</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Ignore Low Variance</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Combine Rare Levels</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Rare Level Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Numeric Binning</td>
      <td>False</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Remove Outliers</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Outliers Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Remove Multicollinearity</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Multicollinearity Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Remove Perfect Collinearity</td>
      <td>True</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Clustering</td>
      <td>False</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Clustering Iteration</td>
      <td>None</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Polynomial Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Polynomial Degree</td>
      <td>None</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Trignometry Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Polynomial Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Group Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Feature Selection</td>
      <td>True</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Feature Selection Method</td>
      <td>classic</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Features Selection Threshold</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Feature Interaction</td>
      <td>False</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Feature Ratio</td>
      <td>False</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Interaction Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Fix Imbalance</td>
      <td>False</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Fix Imbalance Method</td>
      <td>SMOTE</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a2451dab-0bdb-4a90-a7b1-33e23d5f9739')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a2451dab-0bdb-4a90-a7b1-33e23d5f9739 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a2451dab-0bdb-4a90-a7b1-33e23d5f9739');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



- 위에 다 되면 빈칸 enter~!!


## 모델링


```python
best_model = compare_models()
```



  <div id="df-98b64b77-316c-420f-86bd-7e4053c4297c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
      <th>TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ridge</th>
      <td>Ridge Classifier</td>
      <td>0.8254</td>
      <td>0.0000</td>
      <td>0.3637</td>
      <td>0.6913</td>
      <td>0.4764</td>
      <td>0.3836</td>
      <td>0.4122</td>
      <td>0.048</td>
    </tr>
    <tr>
      <th>lda</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.8247</td>
      <td>0.7634</td>
      <td>0.3755</td>
      <td>0.6794</td>
      <td>0.4835</td>
      <td>0.3884</td>
      <td>0.4132</td>
      <td>0.288</td>
    </tr>
    <tr>
      <th>gbc</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.8226</td>
      <td>0.7789</td>
      <td>0.3551</td>
      <td>0.6806</td>
      <td>0.4664</td>
      <td>0.3725</td>
      <td>0.4010</td>
      <td>6.496</td>
    </tr>
    <tr>
      <th>ada</th>
      <td>Ada Boost Classifier</td>
      <td>0.8221</td>
      <td>0.7697</td>
      <td>0.3505</td>
      <td>0.6811</td>
      <td>0.4626</td>
      <td>0.3690</td>
      <td>0.3983</td>
      <td>1.317</td>
    </tr>
    <tr>
      <th>lightgbm</th>
      <td>Light Gradient Boosting Machine</td>
      <td>0.8210</td>
      <td>0.7750</td>
      <td>0.3609</td>
      <td>0.6679</td>
      <td>0.4683</td>
      <td>0.3721</td>
      <td>0.3977</td>
      <td>0.436</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>Random Forest Classifier</td>
      <td>0.8199</td>
      <td>0.7598</td>
      <td>0.3663</td>
      <td>0.6601</td>
      <td>0.4707</td>
      <td>0.3727</td>
      <td>0.3965</td>
      <td>3.086</td>
    </tr>
    <tr>
      <th>et</th>
      <td>Extra Trees Classifier</td>
      <td>0.8092</td>
      <td>0.7377</td>
      <td>0.3677</td>
      <td>0.6047</td>
      <td>0.4571</td>
      <td>0.3497</td>
      <td>0.3657</td>
      <td>2.374</td>
    </tr>
    <tr>
      <th>lr</th>
      <td>Logistic Regression</td>
      <td>0.7814</td>
      <td>0.6410</td>
      <td>0.0003</td>
      <td>0.1000</td>
      <td>0.0006</td>
      <td>0.0003</td>
      <td>0.0034</td>
      <td>1.165</td>
    </tr>
    <tr>
      <th>dummy</th>
      <td>Dummy Classifier</td>
      <td>0.7814</td>
      <td>0.5000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.032</td>
    </tr>
    <tr>
      <th>knn</th>
      <td>K Neighbors Classifier</td>
      <td>0.7547</td>
      <td>0.5939</td>
      <td>0.1763</td>
      <td>0.3719</td>
      <td>0.2388</td>
      <td>0.1145</td>
      <td>0.1259</td>
      <td>0.992</td>
    </tr>
    <tr>
      <th>dt</th>
      <td>Decision Tree Classifier</td>
      <td>0.7293</td>
      <td>0.6147</td>
      <td>0.4104</td>
      <td>0.3878</td>
      <td>0.3986</td>
      <td>0.2242</td>
      <td>0.2245</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>svm</th>
      <td>SVM - Linear Kernel</td>
      <td>0.7277</td>
      <td>0.0000</td>
      <td>0.1017</td>
      <td>0.1671</td>
      <td>0.0984</td>
      <td>0.0067</td>
      <td>0.0075</td>
      <td>0.493</td>
    </tr>
    <tr>
      <th>qda</th>
      <td>Quadratic Discriminant Analysis</td>
      <td>0.5098</td>
      <td>0.5473</td>
      <td>0.6141</td>
      <td>0.2472</td>
      <td>0.3488</td>
      <td>0.0600</td>
      <td>0.0805</td>
      <td>0.158</td>
    </tr>
    <tr>
      <th>nb</th>
      <td>Naive Bayes</td>
      <td>0.3760</td>
      <td>0.6442</td>
      <td>0.8845</td>
      <td>0.2441</td>
      <td>0.3826</td>
      <td>0.0608</td>
      <td>0.1207</td>
      <td>0.050</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-98b64b77-316c-420f-86bd-7e4053c4297c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-98b64b77-316c-420f-86bd-7e4053c4297c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-98b64b77-316c-420f-86bd-7e4053c4297c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



- 가장 좋은 모델을 뽑아주세요!


```python
print(best_model)
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>



    RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=123, solver='auto',
                    tol=0.001)
    

- 모델 확인


```python
models()
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>







  <div id="df-fbacfacd-6656-4bca-ba91-838dd08cdeb1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Reference</th>
      <th>Turbo</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lr</th>
      <td>Logistic Regression</td>
      <td>sklearn.linear_model._logistic.LogisticRegression</td>
      <td>True</td>
    </tr>
    <tr>
      <th>knn</th>
      <td>K Neighbors Classifier</td>
      <td>sklearn.neighbors._classification.KNeighborsCl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>nb</th>
      <td>Naive Bayes</td>
      <td>sklearn.naive_bayes.GaussianNB</td>
      <td>True</td>
    </tr>
    <tr>
      <th>dt</th>
      <td>Decision Tree Classifier</td>
      <td>sklearn.tree._classes.DecisionTreeClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>svm</th>
      <td>SVM - Linear Kernel</td>
      <td>sklearn.linear_model._stochastic_gradient.SGDC...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>rbfsvm</th>
      <td>SVM - Radial Kernel</td>
      <td>sklearn.svm._classes.SVC</td>
      <td>False</td>
    </tr>
    <tr>
      <th>gpc</th>
      <td>Gaussian Process Classifier</td>
      <td>sklearn.gaussian_process._gpc.GaussianProcessC...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mlp</th>
      <td>MLP Classifier</td>
      <td>sklearn.neural_network._multilayer_perceptron....</td>
      <td>False</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>Ridge Classifier</td>
      <td>sklearn.linear_model._ridge.RidgeClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>Random Forest Classifier</td>
      <td>sklearn.ensemble._forest.RandomForestClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>qda</th>
      <td>Quadratic Discriminant Analysis</td>
      <td>sklearn.discriminant_analysis.QuadraticDiscrim...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>ada</th>
      <td>Ada Boost Classifier</td>
      <td>sklearn.ensemble._weight_boosting.AdaBoostClas...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>gbc</th>
      <td>Gradient Boosting Classifier</td>
      <td>sklearn.ensemble._gb.GradientBoostingClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>lda</th>
      <td>Linear Discriminant Analysis</td>
      <td>sklearn.discriminant_analysis.LinearDiscrimina...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>et</th>
      <td>Extra Trees Classifier</td>
      <td>sklearn.ensemble._forest.ExtraTreesClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>lightgbm</th>
      <td>Light Gradient Boosting Machine</td>
      <td>lightgbm.sklearn.LGBMClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>dummy</th>
      <td>Dummy Classifier</td>
      <td>sklearn.dummy.DummyClassifier</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fbacfacd-6656-4bca-ba91-838dd08cdeb1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fbacfacd-6656-4bca-ba91-838dd08cdeb1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fbacfacd-6656-4bca-ba91-838dd08cdeb1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 모델 생성 (knn)


```python
knn_model = create_model('knn')
```



  <div id="df-1188fb14-1614-4158-a810-7cfea2e706c8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
    <tr>
      <th>Fold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7531</td>
      <td>0.6064</td>
      <td>0.1920</td>
      <td>0.3743</td>
      <td>0.2538</td>
      <td>0.1239</td>
      <td>0.1338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7544</td>
      <td>0.5927</td>
      <td>0.1977</td>
      <td>0.3812</td>
      <td>0.2604</td>
      <td>0.1305</td>
      <td>0.1406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.7500</td>
      <td>0.5850</td>
      <td>0.1461</td>
      <td>0.3355</td>
      <td>0.2036</td>
      <td>0.0818</td>
      <td>0.0917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.7431</td>
      <td>0.5754</td>
      <td>0.1519</td>
      <td>0.3174</td>
      <td>0.2054</td>
      <td>0.0744</td>
      <td>0.0816</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.7556</td>
      <td>0.5796</td>
      <td>0.1576</td>
      <td>0.3642</td>
      <td>0.2200</td>
      <td>0.1013</td>
      <td>0.1138</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.7538</td>
      <td>0.5969</td>
      <td>0.1748</td>
      <td>0.3675</td>
      <td>0.2369</td>
      <td>0.1117</td>
      <td>0.1227</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.7600</td>
      <td>0.5929</td>
      <td>0.1891</td>
      <td>0.3976</td>
      <td>0.2563</td>
      <td>0.1343</td>
      <td>0.1475</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7613</td>
      <td>0.6223</td>
      <td>0.1719</td>
      <td>0.3947</td>
      <td>0.2395</td>
      <td>0.1232</td>
      <td>0.1382</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.7688</td>
      <td>0.5926</td>
      <td>0.1948</td>
      <td>0.4359</td>
      <td>0.2693</td>
      <td>0.1552</td>
      <td>0.1730</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.7542</td>
      <td>0.5708</td>
      <td>0.1695</td>
      <td>0.3642</td>
      <td>0.2314</td>
      <td>0.1077</td>
      <td>0.1189</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.7554</td>
      <td>0.5915</td>
      <td>0.1745</td>
      <td>0.3733</td>
      <td>0.2377</td>
      <td>0.1144</td>
      <td>0.1262</td>
    </tr>
    <tr>
      <th>Std</th>
      <td>0.0065</td>
      <td>0.0144</td>
      <td>0.0176</td>
      <td>0.0313</td>
      <td>0.0216</td>
      <td>0.0232</td>
      <td>0.0254</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1188fb14-1614-4158-a810-7cfea2e706c8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1188fb14-1614-4158-a810-7cfea2e706c8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1188fb14-1614-4158-a810-7cfea2e706c8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




```python
# 하이퍼파라미터 여기에서 튜닝
import numpy as np
params ={
    'n_neighbors' : np.arange(0, 50, 1)
}
tunned_knn = tune_model(knn_model, custom_grid=params)
print(tunned_knn)
```



  <div id="df-8bbeba7c-01ad-4759-978b-ef4b82675912">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
    <tr>
      <th>Fold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7813</td>
      <td>0.6482</td>
      <td>0.0372</td>
      <td>0.5000</td>
      <td>0.0693</td>
      <td>0.0402</td>
      <td>0.0876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7807</td>
      <td>0.6436</td>
      <td>0.0315</td>
      <td>0.4783</td>
      <td>0.0591</td>
      <td>0.0330</td>
      <td>0.0759</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.7744</td>
      <td>0.6563</td>
      <td>0.0315</td>
      <td>0.3333</td>
      <td>0.0576</td>
      <td>0.0206</td>
      <td>0.0403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.7845</td>
      <td>0.6589</td>
      <td>0.0659</td>
      <td>0.5610</td>
      <td>0.1179</td>
      <td>0.0754</td>
      <td>0.1345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.7826</td>
      <td>0.6645</td>
      <td>0.0315</td>
      <td>0.5500</td>
      <td>0.0596</td>
      <td>0.0368</td>
      <td>0.0903</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.7794</td>
      <td>0.6477</td>
      <td>0.0544</td>
      <td>0.4634</td>
      <td>0.0974</td>
      <td>0.0539</td>
      <td>0.0961</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.7826</td>
      <td>0.6278</td>
      <td>0.0630</td>
      <td>0.5238</td>
      <td>0.1125</td>
      <td>0.0688</td>
      <td>0.1214</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7751</td>
      <td>0.6702</td>
      <td>0.0372</td>
      <td>0.3611</td>
      <td>0.0675</td>
      <td>0.0278</td>
      <td>0.0523</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.7813</td>
      <td>0.6409</td>
      <td>0.0630</td>
      <td>0.5000</td>
      <td>0.1120</td>
      <td>0.0662</td>
      <td>0.1146</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.7881</td>
      <td>0.6426</td>
      <td>0.0661</td>
      <td>0.6389</td>
      <td>0.1198</td>
      <td>0.0822</td>
      <td>0.1548</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.7810</td>
      <td>0.6501</td>
      <td>0.0482</td>
      <td>0.4910</td>
      <td>0.0873</td>
      <td>0.0505</td>
      <td>0.0968</td>
    </tr>
    <tr>
      <th>Std</th>
      <td>0.0039</td>
      <td>0.0119</td>
      <td>0.0148</td>
      <td>0.0861</td>
      <td>0.0255</td>
      <td>0.0206</td>
      <td>0.0338</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8bbeba7c-01ad-4759-978b-ef4b82675912')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8bbeba7c-01ad-4759-978b-ef4b82675912 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8bbeba7c-01ad-4759-978b-ef4b82675912');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=-1, n_neighbors=42, p=2,
                         weights='uniform')
    

## 결론 : 데이터 분석은 기획력이 중요하다.
- 어떻게 서비스 할 것이냐.
- 분석을 하는 개발자


```python
# 시각화
plot_model(knn_model, plot='auc')
```


    
![png](output_22_0.png)
    


### auc
- 최소 0.5
- 좋은 모델 기준 0.8
- 최대 1.0


```python
plot_model(knn_model, plot='confusion_matrix')
```


    
![png](output_24_0.png)
    



```python
evaluate_model(tunned_knn)
```



<script src="/static/components/requirejs/require.js"></script>
<script>
  requirejs.config({
    paths: {
      base: '/static/base',
      plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
    },
  });
</script>




    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…


## 어떤 모델을 사용할 것이냐를 빠르게 판단
- 정확한 데이터는 셋팅을 해주는 것이 좋다.
