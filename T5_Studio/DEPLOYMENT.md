# Despliegue de T5 Studio en Streamlit Community Cloud

Esta guía permite publicar la aplicación para que cualquier persona pueda probarla desde un navegador, sin instalar Python ni dependencias localmente.

## 1. Crear el repositorio

1. Inicia sesión en GitHub.
2. Selecciona **New repository**.
3. Nombre sugerido: `t5-studio-inferencia`.
4. Selecciona **Public** si deseas que la docente y cualquier persona puedan revisar el código sin solicitar acceso.
5. No agregues README, `.gitignore` ni licencia desde GitHub, porque ya están incluidos en este proyecto.
6. Crea el repositorio.

## 2. Subir los archivos

En el repositorio recién creado:

1. Selecciona **uploading an existing file** o **Add file → Upload files**.
2. Sube el contenido de esta carpeta conservando esta estructura:

```text
T5-Studio/
├── app.py
├── README.md
├── requirements.txt
├── DEPLOYMENT.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── T5_Small_Inferencia_V2.ipynb
└── assets/
    ├── captura_1.png
    ├── captura_2.png
    ├── captura_3.png
    ├── captura_4.png
    └── captura_5.png
```

3. Escribe un mensaje como `Entrega final T5 Studio`.
4. Selecciona **Commit changes**.

## 3. Publicar en Streamlit Community Cloud

1. Ve a https://share.streamlit.io/
2. Inicia sesión con GitHub.
3. Selecciona **Create app**.
4. Elige el repositorio creado.
5. Usa la rama `main`.
6. En **Main file path**, escribe `app.py`.
7. Elige un subdominio disponible, por ejemplo `t5-studio-inferencia`.
8. Selecciona **Deploy**.

La primera carga puede tardar porque Streamlit debe instalar las dependencias y descargar `google-t5/t5-small`.

## 4. Probar el despliegue

Cuando finalice, Streamlit mostrará una URL semejante a:

```text
https://t5-studio-inferencia.streamlit.app
```

Abre la URL en una ventana de incógnito para verificar que pueda usarse sin iniciar sesión.

Prueba al menos:

- **Resumen** con el texto precargado.
- **Traducción EN → DE** con `Artificial intelligence can support better decisions.`

En Streamlit Community Cloud es normal que el dispositivo muestre `cpu`. Las capturas del trabajo fueron realizadas en un entorno con CUDA y, por ello, muestran `cuda`.

## 5. Agregar el enlace público al README

Una vez tengas la URL definitiva, edita `README.md` en GitHub y reemplaza:

```text
_agregar el enlace después del despliegue_
```

por tu URL real. Después selecciona **Commit changes**.

## 6. Qué debe entregar al docente

- URL pública del repositorio GitHub.
- URL pública de Streamlit.
- PDF solicitado con entradas, salidas, interpretación de Q/K/V y capturas propias del despliegue.

De esta manera, el docente puede revisar tanto el código como la aplicación ejecutable.
