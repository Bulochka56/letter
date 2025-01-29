Letter - Web-приложение искусственного интеллекта. Приложение может быть использования для распознавания письменного текста с изображения.

Для распознавания изображений используется нейронная сеть: https://huggingface.co/kazars24/trocr-base-handwritten-ru

Интерфейс реализован на базе Streamlit.

Для подготовки среды выполнения:

1 Установить python 3.9

2 создать виртуальную среду: 
    py -3.9 -m venv venv  

3 активировать среду: 
    .venv\scripts\activate.bat(activate.ps1)

4 Установить зависимости:
    pip install requirements.txt

Для локального запуска ввести в коммандной строке:
streamlit run letter.py