# Cifras de población e intérpretes tras la respuesta del INSOR (N-56)

Párrafos de reemplazo para el **documento final de tesis**. El anteproyecto aprobado no se modifica.
Fuente nueva: oficio n.º 20263700000291 del INSOR (30-sep-2026), respuesta al radicado 20262800003029.
El PDF está en `Tesis/Gestiones/0b_INSOR/`, fuera del repositorio porque contiene datos personales.

La respuesta cambia cuatro cosas:

1. **Intérpretes en el RENI:** hay cifra oficial con fecha de corte, 71 + 1 al 25-sep-2026. Además, el
   INSOR aclara que el listado público solo incluye a quienes autorizaron publicar sus datos.
2. **Los «400 intérpretes»:** el INSOR no reconoce la fuente. Sus caracterizaciones voluntarias dan
   668 (2020) y 465 (2025).
3. **Población:** ninguna de las cifras de población mide el uso de la LSC, y el INSOR pide citar fuente,
   año y criterio.
4. **Caída del RLCPD de 2020 a 2023:** queda explicada por el cambio de procedimiento (Resolución 113
   de 2020).

Están escritos en LaTeX con la numeración de referencias del anteproyecto. `refINSOR26` es una
referencia nueva; la entrada propuesta está al final.

---

## 1. Resumen (reemplaza `02_resumen.tex:31-35`)

```latex
La motivación parte de una necesidad social documentada, aunque su tamaño no puede fijarse con una
sola cifra. Las fuentes oficiales miden poblaciones distintas: 314.320 personas reportaron no poder
oír o hacerlo con mucha dificultad en el Censo de 2018 \cite{ref37}, y los registros y encuestas
de discapacidad arrojan otras magnitudes según su método \cite{ref37,ref39,ref44}. Ninguna mide
cuántas personas usan la Lengua de Señas Colombiana, como precisa el propio INSOR \cite{refINSOR26}.
Frente a esa población, la oferta de interpretación con reconocimiento oficial es reducida: el
```

*(La frase siguiente del resumen, sobre la oferta de interpretación, continúa igual o con la cifra
71 + 1 de la sección 3.)*

---

## 2. Planteamiento del problema: población (reemplaza `03_problema.tex:22-37`)

```latex
La comunicación entre la comunidad sorda y la comunidad oyente en Colombia enfrenta una brecha
estructural, pero el tamaño de la población afectada depende del instrumento con que se mida. El
Censo Nacional de Población y Vivienda de 2018 registra \textbf{314.320} personas que reportaron no
poder oír la voz o los sonidos, o hacerlo con mucha dificultad \cite{ref37}; una medición referida
a la Encuesta Nacional de Calidad de Vida sitúa la cifra en \textbf{459.784} personas \cite{ref44};
y el Registro de Localización y Caracterización de Personas con Discapacidad (RLCPD) contabilizaba
\textbf{166.619} personas en 2020 \cite{ref37} y \textbf{29.038} en 2023 \cite{ref39}. Consultado
formalmente por este proyecto, el INSOR precisó que estas cifras responden a objetivos,
metodologías, períodos y poblaciones de referencia distintos y no son directamente comparables, y
que es necesario diferenciar entre personas con dificultad para oír, personas con discapacidad
auditiva y personas usuarias de la LSC: una dificultad para oír no implica el uso de la lengua
\cite{refINSOR26}. La diferencia entre los dos cortes del RLCPD no indica una disminución de la
población: hasta el 30 de junio de 2020 el registro se nutría del autorreporte, y desde noviembre
de 2020, en aplicación de la Resolución 113 de 2020, el ingreso exige un certificado de
discapacidad expedido tras una valoración multidisciplinaria, de modo que el registro nuevo se
puebla de forma gradual \cite{refINSOR26}. Con la información disponible no es posible estimar
cuántas personas usan la LSC en Colombia \cite{refINSOR26}; por eso este documento cita cada cifra
con su fuente, su año y el criterio con que identifica a la población, y no presenta ninguna como
el número de usuarios de la lengua.
```

---

## 3. Planteamiento del problema: intérpretes (reemplaza `03_problema.tex:39-66`)

```latex
Frente a esa población, la oferta de interpretación con reconocimiento oficial es reducida. El
oficio se profesionalizó mediante la Resolución 10185 de 2018 del Ministerio de Educación Nacional
\cite{ref40}, que reglamenta el reconocimiento de intérpretes oficiales y deroga la Resolución
5274 de 2017; el registro correspondiente se rige por la Resolución 045 de 2022 del INSOR y por el
artículo 7 de la Ley 982 de 2005 \cite{ref31}. Con corte al 25 de septiembre de 2026, el Registro
Nacional de Intérpretes de Lengua de Señas Colombiana--Español y Guías Intérpretes (RENI) publica
\textbf{71 intérpretes de LSC--español y 1 guía intérprete}, con presencia en 20 departamentos y 6
registros sin departamento declarado \cite{refINSOR26,ref47}. El listado público no es el total
de inscritos: incluye solo a quienes, además de cumplir los requisitos, autorizaron la publicación
de sus datos \cite{refINSOR26}. Tampoco mide el ejercicio de la interpretación, porque el
reconocimiento oficial no es requisito para ejercerla \cite{ref46}. Para esa magnitud, el INSOR
dispone de dos caracterizaciones propias por encuesta voluntaria ---668 registros en 2020 y
\textbf{465} en 2025---, que no son censos ni muestras y no deben leerse como el número total de
intérpretes del país \cite{refINSOR26}. La cifra de «400 intérpretes» que publicó \emph{El Tiempo}
en 2020 \cite{ref45}, y que la exposición de motivos del Proyecto de Ley 400 de 2025 Cámara
atribuye al RENI \cite{ref46}, no tiene una fuente identificada por el INSOR y no se adopta en
este documento \cite{refINSOR26}. Ninguna de estas cifras discute el orden de la desproporción:
unas decenas de intérpretes con reconocimiento oficial publicados, o unos centenares de personas
que se reconocen como intérpretes en las caracterizaciones voluntarias, frente a cientos de miles
de personas que reportan dificultad para oír.
```

**Cambios respecto del anteproyecto:**
- Se reemplaza el conteo propio del 8-sep (70 + 1, contando páginas del listado) por la cifra oficial.
  Si se quiere conservar el conteo propio como comprobación, puede ir en una nota al pie: con 70 + 1 al
  8-sep y 71 + 1 al 25-sep, la diferencia es de un registro entre las dos fechas.
- Se retira «a partir de cifras manejadas por el INSOR» para los 400, porque el INSOR no reconoce la fuente.
- La «subinscripción», que antes se infería del artículo 14 del proyecto de ley, ahora tiene respaldo
  directo del INSOR.

---

## 4. Justificación, utilidad (reemplaza `05_justificacion.tex:6-10`)

```latex
El proyecto resulta pertinente porque atiende una necesidad concreta en un marco realista. La
oferta de interpretación con reconocimiento oficial es reducida ---71 intérpretes de LSC y 1 guía
intérprete en el listado público del RENI al 25 de septiembre de 2026, y 465 intérpretes en la
caracterización voluntaria del INSOR de 2025, cifras que no miden lo mismo \cite{refINSOR26}---
frente a una población con dificultades auditivas que las fuentes oficiales cuentan en cientos de
miles de personas según el instrumento (\S1.1). En ese contexto, un prototipo de reconocimiento
limitado a un vocabulario cerrado de 50 señas ofrece una herramienta de apoyo para escenarios
controlados específicos: aulas y entornos
```

---

## 5. Otros lugares que conviene revisar

- **`03_problema.tex:70-74`** (estudiantes sordos según el SIMAT): ya evita calcular porcentajes sobre
  la población, en línea con lo que pide el INSOR. Solo hay que verificar que no diga «población sorda»
  sin nombrar el instrumento.
- **`14_anexos.tex:340-420`** (tabla de fuentes): agregar una fila para el oficio del INSOR y anotar
  que la cifra de 400 intérpretes no tiene fuente reconocida.
- **El texto en el que se dice que la conciliación «se remite a la consulta formal al INSOR prevista en
  la actividad 0.b»:** cambiarlo a pasado. La consulta ya se hizo y su resultado está en este documento.

---

## 6. Entrada bibliográfica propuesta (formato IEEE, como `13_referencias.tex`)

```latex
\bibitem{refINSOR26}
Instituto Nacional para Sordos (INSOR), Subdirección de Promoción y Desarrollo, «Respuesta radicado
20262800003029», oficio n.º 20263700000291, Bogotá, Colombia, 30 de sep. de 2026. Respuesta a
petición de información del autor (Ley 1755 de 2015).
```

Antes de citarla, hay que confirmar con el director cómo se citan las comunicaciones oficiales en el
documento de tesis: como referencia o como anexo, y si se adjunta el oficio sin los datos personales.
