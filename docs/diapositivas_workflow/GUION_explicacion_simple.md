# Guion para presentar `explicacion_simple.pdf`

Qué decir en cada diapositiva, en palabras sencillas. Unos 15 minutos en total.
La versión técnica completa (`workflow_lsc.pdf`, 48 páginas) queda de respaldo para preguntas del jurado.

---

**1. Portada.**
«Voy a mostrar cómo el sistema pasa de una seña hecha frente a la cámara a una palabra dicha en voz alta».

**2. Qué hace el sistema.**
Una persona hace una seña, la cámara la graba, el computador la reconoce y la dice en español. Reconoce 50 señas. Funciona en un portátil normal, sin internet ni tarjeta gráfica.

**3. El camino en 5 pasos.**
Es el mapa de toda la presentación. Los pasos 1 a 3 convierten el video en números. El paso 4, la red neuronal, interpreta esos números. El paso 5 dice la seña en voz alta.

**4. Los datos.**
Aprendimos de un corpus público de la Universidad de los Andes. Son 50 señas, hechas por 5 personas, 4 veces cada una: 1.000 videos. Al revisarlos encontramos 3 videos repetidos, así que quedan 997. Cada seña tiene 20 videos (tres tienen 19), así que las clases están balanceadas.

**5. El computador no mira la imagen, mira puntos.**
MediaPipe, una herramienta de Google, encuentra 21 puntos en cada mano: la muñeca y 4 por dedo. Cada punto son 3 números: horizontal, vertical y profundidad. Una mano son 63 números.
*Si preguntan:* MediaPipe no lo entrenamos nosotros; se usa tal cual.

**6. Se hace en todos los cuadros.**
El video va a 24 cuadros por segundo y se procesan todos. El video de «hola» tiene 135 cuadros, así que sale una tabla de 135 filas.
**Lo más importante de esta diapositiva:** una muestra es el video completo, una seña, no cada cuadro. Por eso no hay desbalance aunque unos videos duren más que otros.

**7. No importa dónde esté la persona.**
Todas las posiciones se miden desde el centro de los hombros y en «anchos de hombros». Así da igual si la persona está más cerca, más lejos o más a un lado de la cámara.

**8. Todas las señas quedan del mismo largo.**
Las señas duran distinto. «Hola» puede tener 135 cuadros o 69, según la repetición. Siempre se toman 30 momentos repartidos del inicio al final. Si un momento cae entre dos cuadros, se calcula el punto intermedio: eso es la interpolación lineal. Es como estirar o encoger un dibujo: cambia el largo, no la forma.

**9. Lo que recibe la red.**
En cada uno de los 30 pasos hay 252 números:
- 126 de posición: los 42 puntos de las dos manos, por 3 números cada uno.
- 126 de velocidad: cuánto se movió cada punto desde el paso anterior.

En total son 7.560 números por seña.

**10. La red lee la seña en orden.**
La red hace tres cosas:
1. Detecta movimientos cortos.
2. Recuerda el orden hacia adelante y hacia atrás. Esa es la **LSTM bidireccional**: sí es una LSTM.
3. Se fija en los momentos clave.

Al final da 50 probabilidades, una por seña. Está hecha con TensorFlow y Keras.

**11. Cómo aprende.**
Se le muestra una seña, adivina, se mide el error y se corrige un poco. Eso se repite miles de veces; cada vuelta completa por los datos es una época. Los videos se separan en tres grupos:
- **Aprender:** con ellos practica.
- **Ajustar:** sirven para saber cuándo parar.
- **Examen final:** la red nunca los vio; de ahí salen los resultados.

**12. Decide y habla.**
Si la probabilidad más alta pasa el umbral, que hoy es 20 %, el sistema dice la seña en voz alta. Si no, pide que se repita.

**13. Qué tan bien funciona.**
- Con personas que ya conocía acierta el 88 %.
- Con personas nuevas acierta el 58 % a la primera, y el 77 % si se cuentan sus tres primeras opciones.

La cifra importante es la de personas nuevas, porque así le iría con alguien real.

**14. Cómo se probó con personas nuevas.**
Son 5 rondas. En cada una se deja a una persona fuera del aprendizaje y se usa como examen. El acierto va del 46 % al 69 % según la persona, y el promedio es 58 %.
*Si preguntan:* esto es una validación cruzada, un «universo» por persona.

**15. Dónde se equivoca.**
Se confunde con señas que se parecen de verdad: niño y niña, ¿cómo estás? y ¿cómo?. «Más o menos» no se acertó ninguna vez: el sistema dice «poco» o «yo». Los errores no son al azar.

**16. ¿Hacía falta la red?**
Sí. Los métodos más simples solo ven un resumen de la seña, sin su orden, y aciertan menos: 42 %, 26 % y 25 %, frente a 58 % de la red. La diferencia es estadísticamente significativa.

**17. Qué tan rápido.**
La persona tiene 2,5 segundos para hacer la seña. Después, el sistema tarda 36 milésimas en responder. Casi toda la espera es el tiempo de hacer la seña.

**18. Lo que sigue.**
Falta la prueba con una persona real: el profesor de LSC vinculado aceptó ayudar. Se hará en condiciones parecidas a las del corpus: unos 2 metros de la cámara, fondo liso y buena luz. Antes hay que acordar con el director cómo se hace esa prueba y qué umbral se usa.

---

## Preguntas probables y respuesta corta

- **¿Cada cuadro es una muestra?** No. La muestra es el video completo de una seña, llevado a 30 pasos.
- **¿Es una LSTM?** Sí, una LSTM bidireccional, con capas Conv1D antes y atención después.
- **¿Cuántas características de entrada tiene?** 252 por paso, en 30 pasos: 21 × 3 × 2 = 126 de posición, más 126 de velocidad.
- **¿Qué se entrena?** Solo la red neuronal, que tiene 714.291 pesos. MediaPipe no se entrena.
- **¿Por qué 58 % y no más?** El corpus solo tiene 5 personas: con tan pocas, el modelo aprende rasgos de cada persona. Por eso con gente conocida llega al 88 %.
