# Python library template

EDAS V0.8

[![GitHub Actions Status](https://github.com/milser/python-library-template/workflows/CI/CD/badge.svg)](https://github.com/milser/python-library-template/actions)
[![Code Coverage](https://codecov.io/gh/milser/Python-library-template/graph/badge.svg?branch=master)](https://codecov.io/gh/milser/Python-library-template?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/milser/Python-library-template.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/milser/Python-library-template/latest/files/)
[![CodeFactor](https://www.codefactor.io/repository/github/milser/Python-library-template/badge)](https://www.codefactor.io/repository/github/milser/Python-library-template)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Docs](https://img.shields.io/badge/docs-master-blue.svg)](https://milser.github.io/python-library-template)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/milser/python-library-template/master)

<!-- Here edas_tatmil should be replaced with your library's name on PyPI  -->
[![PyPI version](https://badge.fury.io/py/edas_tatmil.svg)](https://badge.fury.io/py/edas_tatmil)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/edas_tatmil.svg)](https://pypi.org/project/edas_tatmil/)

The template library is [`edas_tatmil`](https://github.com/milser/Python-library-template/search?q=edas_tatmil&unscoped_q=edas_tatmil) to make it clear what is needed for replacement

## Descripción General  
Esta librería proporciona herramientas útiles para llevar a cabo un EDA completo.  

> **⚠️ Nota:** Este módulo está en fase de desarrollo y puede tener errores.  

> **⚠️ Nota:** Algunas funciones sólo aplican a casos muy generales. Para hacer un buen EDA se debe comprender el caso específico en el que se trabaja y, en muchas ocasiones, se necesitarán acciones que no están recogidas en este módulo.  

> **⚠️ Nota:** Lea detenidamente la descripción de las funciones, ya que algunas necesitan un sistema específico de archivos. La estructura final de archivos si se hace el EDA con edastatmil-milser se muestra en la figura [ffinal](#fig:ffinal).

## Instalación
1. **Requerimientos:**  
   - tabulate  
   - pandas  
   - matplotlib.pyplot  
   - seaborn  
   - math  
   - os  
   - sklearn.model_selection  
   - importlib  
   Pueden instalarse desde la terminal con `pip install`.

2. **Instalar la librería:**  
   ```bash
   pip install edastatmil-milser

3. **Importar lalibrería:**  
   ```bash
   from edastatmil_milser import edas_tatmil as EDA

4. **Ejemplo de llamada de función**
   ```bash
   EDA.function_example

## Funciones
### `get_column_type(series)`

Esta función estudia si una característica es numérica o categórica.

- **Atributos:** no requerido.

- **Ejemplo de uso:**
  ```python
  variables = pd.DataFrame({'Data Type': data_frame.dtypes})
  variables['Data category'] = df.apply(EDA.get_column_type)

Añade una columna nueva al dataframe 'variables' llamada 'Data type' indicando si la variable es categorica o numérica.
- **Return:**
   'Categorical' si la variable es categórica 'Numerical' si es numérica.

## Controlling the version number with bumpversion

When you want to increment the version number for a new release use [`bumpversion`](https://github.com/peritus/bumpversion) to do it correctly across the whole library.
For example, to increment to a new patch release you would simply run

```
bumpversion patch
```

which given the [`.bumpversion.cfg`](https://github.com/milser/Python-library-template/blob/master/.bumpversion.cfg) makes a new commit that increments the release version by one patch release.
