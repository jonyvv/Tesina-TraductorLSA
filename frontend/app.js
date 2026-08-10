/* ==========================================================================
   Traductor LSA — cliente
   --------------------------------------------------------------------------
   Responsabilidades:
     - Capturar video de la cámara y enviar frames al backend por WebSocket.
     - Dibujar el esqueleto de la mano sobre el video.
     - Mostrar la seña reconocida, sus métricas y la latencia del sistema.
     - Acumular el texto traducido.

   Ver docs/ARQUITECTURA.md para el diagrama de secuencia que implementa.
   ========================================================================== */

(() => {
  "use strict";

  /* ======================================================================
     Configuración
     ====================================================================== */

  const WS_URL =
    (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/translate";

  const ANCHO_ENVIO = 480;         // ancho del frame que se manda (px)
  const CALIDAD_JPEG = 0.7;
  const FPS_POR_DEFECTO = 8;

  // Cuántos frames pueden estar "en vuelo" (enviados, sin respuesta) a la vez.
  // Sin este tope, si el backend tarda más que el intervalo de envío, la cola
  // crece sin límite: la latencia sube sin parar y las mediciones dejan de
  // representar el tiempo real de procesamiento. Con 2 se mantiene el pipeline
  // ocupado sin acumular atraso.
  const MAX_FRAMES_EN_VUELO = 2;

  const MUESTRAS_METRICAS = 30;    // ventana para promediar latencia y fps
  const RECONEXION_BASE_MS = 500;  // backoff exponencial: 0.5s, 1s, 2s, 4s...
  const RECONEXION_MAX_MS = 8000;
  const CLAVE_TEMA = "lsa-tema";

  /* ======================================================================
     Referencias del DOM
     ====================================================================== */

  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const overlay = $("overlay");
  const ctxOverlay = overlay.getContext("2d");
  const camaraApagada = $("camara-apagada");
  const hud = $("hud");
  const hudSeña = $("hud-seña");

  const btnIniciar = $("btn-iniciar");
  const btnDetener = $("btn-detener");
  const btnLimpiar = $("btn-limpiar");
  const btnCopiar = $("btn-copiar");
  const btnDeshacer = $("btn-deshacer");
  const btnTema = $("btn-tema");
  const btnTemaIcono = $("btn-tema-icono");

  const estado = $("estado");
  const estadoTexto = $("estado-texto");
  const avisoError = $("aviso-error");

  const barraConfianza = $("barra-confianza");
  const barraConfianzaRelleno = $("barra-confianza-relleno");
  const valorConfianza = $("valor-confianza");
  const barraEstabilidad = $("barra-estabilidad");
  const barraEstabilidadRelleno = $("barra-estabilidad-relleno");
  const valorEstabilidad = $("valor-estabilidad");
  const valorManos = $("valor-manos");

  const valorLatencia = $("valor-latencia");
  const valorServidor = $("valor-servidor");
  const valorRed = $("valor-red");
  const valorFps = $("valor-fps");

  const chkOverlay = $("chk-overlay");
  const chkAutoescribir = $("chk-autoescribir");
  const rangoFps = $("rango-fps");
  const salidaFps = $("salida-fps");

  const textoTraducido = $("texto-traducido");
  const textoVacio = $("texto-vacio");
  const pieConexion = $("pie-conexion");

  /* ======================================================================
     Estado
     ====================================================================== */

  let ws = null;
  let stream = null;
  let temporizadorEnvio = null;
  let temporizadorReconexion = null;
  let intentosReconexion = 0;
  let cerradoPorUsuario = false;

  let conexionesMano = [];          // esqueleto, lo sirve el backend
  let ultimaSeñaConfirmada = null;
  const señasAgregadas = [];

  const enviosPendientes = [];      // timestamps de frames sin respuesta
  const latencias = [];             // round-trip, ms
  const tiemposServidor = [];       // cómputo del backend, ms
  const marcasFps = [];             // timestamps de respuestas recibidas

  const lienzoCaptura = document.createElement("canvas");
  const ctxCaptura = lienzoCaptura.getContext("2d");

  /* ======================================================================
     Tema
     ====================================================================== */

  function aplicarTema(tema) {
    document.documentElement.dataset.tema = tema;
    btnTemaIcono.textContent = tema === "oscuro" ? "☀" : "☾";
    btnTema.setAttribute(
      "aria-label",
      tema === "oscuro" ? "Cambiar a tema claro" : "Cambiar a tema oscuro"
    );
  }

  function alternarTema() {
    const actual =
      document.documentElement.dataset.tema ||
      (matchMedia("(prefers-color-scheme: light)").matches ? "claro" : "oscuro");
    const nuevo = actual === "oscuro" ? "claro" : "oscuro";
    aplicarTema(nuevo);
    try { localStorage.setItem(CLAVE_TEMA, nuevo); } catch { /* modo privado */ }
  }

  function inicializarTema() {
    let guardado = null;
    try { guardado = localStorage.getItem(CLAVE_TEMA); } catch { /* modo privado */ }
    if (guardado === "claro" || guardado === "oscuro") {
      aplicarTema(guardado);
    } else {
      // Sin preferencia guardada seguimos al sistema (lo resuelve el CSS);
      // acá solo se ajusta el ícono del botón.
      const claro = matchMedia("(prefers-color-scheme: light)").matches;
      btnTemaIcono.textContent = claro ? "☾" : "☀";
    }
  }

  /* ======================================================================
     Estado de conexión
     ====================================================================== */

  function setEstado(clase, mensaje) {
    estado.className = "estado" + (clase ? ` estado--${clase}` : "");
    estadoTexto.textContent = mensaje;
  }

  function mostrarError(mensaje) {
    avisoError.textContent = mensaje;
    avisoError.hidden = false;
  }

  function ocultarError() {
    avisoError.hidden = true;
  }

  /* ======================================================================
     Cámara
     ====================================================================== */

  async function iniciarCamara() {
    ocultarError();
    setEstado("conectando", "Solicitando cámara…");

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (err) {
      setEstado("error", "Sin acceso a la cámara");
      mostrarError(
        err && err.name === "NotAllowedError"
          ? "Permiso de cámara denegado. Habilitalo en el navegador y volvé a intentar."
          : "No se pudo acceder a la cámara. Verificá que no esté en uso por otra aplicación."
      );
      console.error("getUserMedia falló:", err);
      return;
    }

    video.srcObject = stream;
    await video.play();

    // Fallback en AMBAS dimensiones: si `videoWidth` todavía es 0, dividir por
    // él da Infinity y el alto del canvas queda en NaN. Un canvas inválido hace
    // que `toBlob` devuelva null y no se envíe ni un frame, sin ningún error.
    const anchoVideo = video.videoWidth || 960;
    const altoVideo = video.videoHeight || 720;

    overlay.width = anchoVideo;
    overlay.height = altoVideo;
    lienzoCaptura.width = ANCHO_ENVIO;
    lienzoCaptura.height = Math.round(ANCHO_ENVIO * (altoVideo / anchoVideo));

    camaraApagada.hidden = true;
    hud.hidden = false;
    btnIniciar.disabled = true;
    btnDetener.disabled = false;

    cerradoPorUsuario = false;
    intentosReconexion = 0;
    conectarWebSocket();
  }

  function detenerCamara() {
    cerradoPorUsuario = true;
    clearTimeout(temporizadorReconexion);
    clearInterval(temporizadorEnvio);

    if (ws) ws.close();
    if (stream) stream.getTracks().forEach((pista) => pista.stop());
    stream = null;
    video.srcObject = null;

    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    camaraApagada.hidden = false;
    hud.hidden = true;
    btnIniciar.disabled = false;
    btnDetener.disabled = true;

    setEstado("", "Desconectado");
    reiniciarMetricas();
  }

  /* ======================================================================
     WebSocket
     ====================================================================== */

  function conectarWebSocket() {
    setEstado("conectando", intentosReconexion ? "Reconectando…" : "Conectando…");

    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      intentosReconexion = 0;
      enviosPendientes.length = 0;
      ocultarError();
      setEstado("conectado", "Conectado");
      reprogramarEnvio();
    };

    ws.onmessage = (evento) => {
      let datos;
      try {
        datos = JSON.parse(evento.data);
      } catch {
        console.error("Respuesta inválida del backend:", evento.data);
        return;
      }
      registrarMetricas(datos);
      actualizarInterfaz(datos);
    };

    ws.onclose = () => {
      clearInterval(temporizadorEnvio);
      enviosPendientes.length = 0;
      if (cerradoPorUsuario) {
        setEstado("", "Desconectado");
        return;
      }
      // La conexión se cayó sola y la cámara sigue prendida: reintentamos en
      // vez de dejar al usuario mirando un video sin ningún feedback.
      programarReconexion();
    };

    ws.onerror = (err) => console.error("Error de WebSocket:", err);
  }

  function programarReconexion() {
    intentosReconexion += 1;
    const espera = Math.min(
      RECONEXION_BASE_MS * 2 ** (intentosReconexion - 1),
      RECONEXION_MAX_MS
    );
    setEstado("conectando", `Reconectando (intento ${intentosReconexion})…`);
    temporizadorReconexion = setTimeout(() => {
      if (!cerradoPorUsuario) conectarWebSocket();
    }, espera);
  }

  function reprogramarEnvio() {
    clearInterval(temporizadorEnvio);
    const fps = Number(rangoFps.value) || FPS_POR_DEFECTO;
    temporizadorEnvio = setInterval(enviarFrame, Math.round(1000 / fps));
  }

  function enviarFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Control de contrapresión: no acumulamos frames que el backend no llegó a
    // procesar (ver MAX_FRAMES_EN_VUELO).
    if (enviosPendientes.length >= MAX_FRAMES_EN_VUELO) return;

    // ESPEJADO — ver ESPEJADO_CANONICO en common/features.py.
    // El `transform: scaleX(-1)` del CSS es SOLO presentación: `drawImage` lee
    // el frame original. Sin este scale(-1, 1), el backend recibiría la imagen
    // con la orientación contraria a la usada al capturar el dataset, MediaPipe
    // asignaría la handedness al revés y la seña caería en el slot equivocado
    // del vector de features. No da error: solo predice mal.
    ctxCaptura.save();
    ctxCaptura.scale(-1, 1);
    ctxCaptura.drawImage(video, -lienzoCaptura.width, 0,
                         lienzoCaptura.width, lienzoCaptura.height);
    ctxCaptura.restore();

    lienzoCaptura.toBlob(
      (blob) => {
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((buffer) => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            enviosPendientes.push(performance.now());
            ws.send(buffer);
          }
        });
      },
      "image/jpeg",
      CALIDAD_JPEG
    );
  }

  /* ======================================================================
     Métricas
     ====================================================================== */

  function registrarMetricas(datos) {
    const enviado = enviosPendientes.shift();
    if (enviado !== undefined) {
      acotar(latencias, performance.now() - enviado);
    }
    if (typeof datos.ms_servidor === "number") {
      acotar(tiemposServidor, datos.ms_servidor);
    }
    acotar(marcasFps, performance.now());
    refrescarMetricas();
  }

  function acotar(lista, valor) {
    lista.push(valor);
    if (lista.length > MUESTRAS_METRICAS) lista.shift();
  }

  function promedio(lista) {
    if (!lista.length) return null;
    return lista.reduce((a, b) => a + b, 0) / lista.length;
  }

  function refrescarMetricas() {
    const latencia = promedio(latencias);
    const servidor = promedio(tiemposServidor);

    valorLatencia.textContent = latencia === null ? "—" : `${latencia.toFixed(0)} ms`;
    valorServidor.textContent = servidor === null ? "—" : `${servidor.toFixed(0)} ms`;

    // Round-trip menos cómputo del servidor = red + codificación/decodificación
    // JPEG. Separarlos permite saber si conviene optimizar el modelo o el
    // transporte (ver docs/ARQUITECTURA.md §6 y §9).
    if (latencia !== null && servidor !== null) {
      valorRed.textContent = `${Math.max(0, latencia - servidor).toFixed(0)} ms`;
    } else {
      valorRed.textContent = "—";
    }

    if (marcasFps.length >= 2) {
      const lapso = (marcasFps[marcasFps.length - 1] - marcasFps[0]) / 1000;
      valorFps.textContent = lapso > 0
        ? `${((marcasFps.length - 1) / lapso).toFixed(1)}`
        : "—";
    } else {
      valorFps.textContent = "—";
    }
  }

  function reiniciarMetricas() {
    latencias.length = 0;
    tiemposServidor.length = 0;
    marcasFps.length = 0;
    enviosPendientes.length = 0;
    refrescarMetricas();
    actualizarBarra(barraConfianza, barraConfianzaRelleno, valorConfianza, null, "Confianza");
    actualizarBarra(barraEstabilidad, barraEstabilidadRelleno, valorEstabilidad, null, "Estabilidad");
    valorManos.textContent = "—";
    hudSeña.textContent = "—";
    ultimaSeñaConfirmada = null;
  }

  /* ======================================================================
     Interfaz
     ====================================================================== */

  function actualizarInterfaz(datos) {
    if (datos.error) {
      setEstado("error", "Error del backend");
      mostrarError(datos.error);
      hudSeña.textContent = "—";
      return;
    }
    ocultarError();
    if (ws && ws.readyState === WebSocket.OPEN) setEstado("conectado", "Conectado");

    const seña = datos["seña"];
    const confianza = datos.confianza ?? 0;
    const estabilidad = datos.estabilidad ?? 0;
    const valida = datos.valida;

    hudSeña.textContent = seña || "—";
    actualizarBarra(barraConfianza, barraConfianzaRelleno, valorConfianza, confianza, "Confianza");
    actualizarBarra(barraEstabilidad, barraEstabilidadRelleno, valorEstabilidad, estabilidad, "Estabilidad");
    valorManos.textContent = datos.manos ?? 0;

    dibujarLandmarks(datos.landmarks || []);

    if (valida && seña && seña !== ultimaSeñaConfirmada) {
      ultimaSeñaConfirmada = seña;
      if (chkAutoescribir.checked) agregarAlTexto(seña);
    }
    // Al soltar la seña se habilita volver a escribir la misma.
    if (!valida) ultimaSeñaConfirmada = null;
  }

  function actualizarBarra(barra, relleno, salida, valor, nombre) {
    if (valor === null || valor === undefined) {
      relleno.style.width = "0%";
      salida.textContent = "—";
      barra.setAttribute("aria-valuenow", "0");
      barra.setAttribute("aria-valuetext", `${nombre}: sin datos`);
      return;
    }
    const porcentaje = Math.round(Math.max(0, Math.min(1, valor)) * 100);
    relleno.style.width = `${porcentaje}%`;
    salida.textContent = `${porcentaje}%`;

    relleno.classList.toggle("barra__relleno--bajo", porcentaje < 40);
    relleno.classList.toggle("barra__relleno--medio", porcentaje >= 40 && porcentaje < 70);

    barra.setAttribute("aria-valuenow", String(porcentaje));
    barra.setAttribute("aria-valuetext", `${nombre}: ${porcentaje} por ciento`);
  }

  /* ======================================================================
     Overlay de landmarks
     ====================================================================== */

  function dibujarLandmarks(manos) {
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    if (!chkOverlay.checked || !manos.length) return;

    const estilos = getComputedStyle(document.documentElement);
    const colorLinea = estilos.getPropertyValue("--overlay-linea").trim() || "#fff";
    const colorPunto = estilos.getPropertyValue("--overlay-punto").trim() || "#4fd1c5";
    const escala = overlay.width / 640; // grosores proporcionales a la resolución

    for (const mano of manos) {
      const puntos = mano.puntos.map(([x, y]) => [x * overlay.width, y * overlay.height]);

      ctxOverlay.lineWidth = Math.max(1.5, 2.5 * escala);
      ctxOverlay.strokeStyle = colorLinea;
      ctxOverlay.lineCap = "round";
      for (const [a, b] of conexionesMano) {
        if (!puntos[a] || !puntos[b]) continue;
        ctxOverlay.beginPath();
        ctxOverlay.moveTo(puntos[a][0], puntos[a][1]);
        ctxOverlay.lineTo(puntos[b][0], puntos[b][1]);
        ctxOverlay.stroke();
      }

      ctxOverlay.fillStyle = colorPunto;
      const radio = Math.max(2.5, 4 * escala);
      for (const [x, y] of puntos) {
        ctxOverlay.beginPath();
        ctxOverlay.arc(x, y, radio, 0, Math.PI * 2);
        ctxOverlay.fill();
      }
    }
  }

  async function cargarConfiguracionDelModelo() {
    // El esqueleto de la mano lo define `common/features.py` y lo sirve el
    // backend: se pide en vez de duplicarlo acá, para que no pueda quedar
    // desincronizado si alguna vez cambia (misma razón por la que existe common/).
    try {
      const respuesta = await fetch("/model/info");
      if (!respuesta.ok) throw new Error(`HTTP ${respuesta.status}`);
      const info = await respuesta.json();
      conexionesMano = info.conexiones_mano || [];
      pieConexion.textContent = info.clases?.length
        ? `${info.clases.length} señas en el modelo · ${WS_URL}`
        : `Sin modelo entrenado · ${WS_URL}`;
    } catch (err) {
      console.warn("No se pudo leer /model/info; el overlay dibujará solo puntos.", err);
      pieConexion.textContent = WS_URL;
    }
  }

  /* ======================================================================
     Texto traducido
     ====================================================================== */

  function agregarAlTexto(seña) {
    textoVacio.hidden = true;
    const nodo = document.createElement("span");
    nodo.className = "seña-agregada";
    // Las letras sueltas se concatenan; las palabras van separadas por espacios.
    nodo.textContent = seña.length === 1 ? seña : `${seña} `;
    textoTraducido.appendChild(nodo);
    señasAgregadas.push(nodo);
    actualizarBotonesTexto();
  }

  function deshacerUltima() {
    const nodo = señasAgregadas.pop();
    if (nodo) nodo.remove();
    if (!señasAgregadas.length) textoVacio.hidden = false;
    ultimaSeñaConfirmada = null;
    actualizarBotonesTexto();
  }

  function limpiarTexto() {
    señasAgregadas.forEach((nodo) => nodo.remove());
    señasAgregadas.length = 0;
    textoVacio.hidden = false;
    ultimaSeñaConfirmada = null;
    actualizarBotonesTexto();
  }

  async function copiarTexto() {
    const texto = señasAgregadas.map((nodo) => nodo.textContent).join("").trim();
    if (!texto) return;
    try {
      await navigator.clipboard.writeText(texto);
      const original = btnCopiar.textContent;
      btnCopiar.textContent = "¡Copiado!";
      setTimeout(() => { btnCopiar.textContent = original; }, 1600);
    } catch (err) {
      mostrarError("No se pudo copiar al portapapeles.");
      console.error(err);
    }
  }

  function actualizarBotonesTexto() {
    const hay = señasAgregadas.length > 0;
    btnCopiar.disabled = !hay;
    btnDeshacer.disabled = !hay;
    btnLimpiar.disabled = !hay;
  }

  /* ======================================================================
     Eventos
     ====================================================================== */

  btnIniciar.addEventListener("click", iniciarCamara);
  btnDetener.addEventListener("click", detenerCamara);
  btnLimpiar.addEventListener("click", limpiarTexto);
  btnDeshacer.addEventListener("click", deshacerUltima);
  btnCopiar.addEventListener("click", copiarTexto);
  btnTema.addEventListener("click", alternarTema);

  rangoFps.addEventListener("input", () => {
    salidaFps.textContent = `${rangoFps.value} fps`;
    if (ws && ws.readyState === WebSocket.OPEN) reprogramarEnvio();
  });

  chkOverlay.addEventListener("change", () => {
    if (!chkOverlay.checked) ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
  });

  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((pista) => pista.stop());
  });

  /* ======================================================================
     Arranque
     ====================================================================== */

  inicializarTema();
  actualizarBotonesTexto();
  salidaFps.textContent = `${rangoFps.value} fps`;
  setEstado("", "Desconectado");
  cargarConfiguracionDelModelo();
})();
