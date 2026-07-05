# Session 11 — Citación verificable a nivel de línea + baseline RAGAS

> **Qué cubre este ejercicio.** La recuperación ya funciona (sesiones 9–10: reformulación,
> híbrida, reranking, expansión, routing multi-índice, decay temporal). El generador de la
> Sesión 9 ya produce una estimación en JSON con citación obligatoria. El problema: esa
> citación era **gruesa y no verificable** (citaba a nivel de estimación global y nada
> garantizaba que la fuente existiera en el contexto recuperado). Aquí (1) subimos la
> citación a **nivel de línea** y la hacemos **verificable programáticamente**, y (2)
> montamos una primera **evaluación objetiva con RAGAS** sobre el golden set.
>
> Todo el código va en inglés (nombres, comentarios, logs, prompts). Esta prosa y el golden
> set van en español.

---

## 1. Qué estamos evaluando exactamente

Dos cosas distintas, con dos herramientas distintas:

| Eje | Pregunta que responde | Cómo se mide |
| --- | --- | --- |
| **Citación verificable** (Parte 1) | ¿Cada línea de la estimación apunta a una fuente que **realmente** estaba en el contexto recuperado? ¿Hay citaciones colgantes (ids inventados)? | `verify_citations()` → `CitationReport` (lógica determinista, sin LLM) |
| **Calidad de generación** (Parte 2) | ¿La estimación es **fiel** a las fuentes, **relevante** a la pregunta, y el **contexto** recuperado es preciso y completo respecto a la referencia experta? | RAGAS (LLM-juez + embeddings) → 4 métricas × 5 consultas |

Son complementarias: la verificación de citaciones es un control **interno y barato** (¿el modelo cita lo que se le dio?); RAGAS es una evaluación **externa y cara** (¿la respuesta es buena comparada con una referencia experta?).

---

## 2. Contexto de arquitectura: del corpus a la estimación citada

El servicio IA (FastAPI) tiene dos mitades que conviene separar: una **ingesta offline** que
llena las tablas vectoriales, y un **pipeline online** que, dada una transcripción, recupera
y genera la estimación. RAGAS evalúa la salida del pipeline online; la verificación de
citaciones se ejecuta como último paso de ese mismo pipeline.

### 2.1 Mini-diagrama del flujo

```
  INGESTA (offline, una vez)                      PIPELINE ONLINE (por consulta)
  ─────────────────────────                       ──────────────────────────────
  data/budgets_sample.json                        transcript (texto libre)
        │  POST /embeddings/ingest                       │
        │  (1 documento por presupuesto)                 ▼
        ▼                                         (1) reformulate_query ........ gpt-5-mini
  chunker estructural (1 chunk/componente)              │   → EstimationQuery (brief estructurado)
        │                                                ▼
        ▼                                         (2) compose_search_text + embed  text-embedding-3-small
  ┌───────────────────────────────┐                     │
  │ budget_chunks (pgvector + FTS) │ ◀───────────  (3) retrieve() ............... híbrida + RRF + reranker
  │ embedding · content_tsv        │   k-NN +            │   (vector + lexical → fusión → cross-encoder)
  └───────────────────────────────┘   lexical           ▼
                                                  (4) truncate_to_token_budget → build_context_block
                                                        │   → <source id=.. document_id=..>…</source> (XML)
                                                        ▼
                                                  (5) generate_estimate ......... gpt-5 (reasoning=high)
                                                        │   → Estimate (módulos→tareas, citación por línea)
                                                        ▼
                                                  (6) verify_citations(estimate, retrieved_ids)
                                                        │   → CitationReport (grounded/dangling/insufficient)
                                                        ▼
                                                  Estimate citado + informe de citaciones
```

Las cuatro entradas que RAGAS necesita salen de este flujo:

| Entrada RAGAS | De dónde sale en el pipeline |
| --- | --- |
| `question` | la `query` del golden set (la petición de estimación) |
| `answer` | el `Estimate` generado en (5), renderizado como texto |
| `contexts` | los chunks que quedaron en el bloque `<source>` tras (4) |
| `ground_truth` | la estimación de referencia experta del golden set extendido |

### 2.2 Ingesta (prerequisito de datos)

El corpus de presupuestos se carga **por HTTP**, un documento por presupuesto, en la tabla
`budget_chunks` (pgvector + columna full-text `content_tsv`). Comando:

```bash
docker compose up -d                                   # stack arriba (la ingesta habla por HTTP)
docker compose run --rm estimator python scripts/query_examples.py
```

Verificación (esperado: 60 chunks de 17 presupuestos):

```bash
docker compose exec -T estimator-postgres psql -U estimator -d estimator -t \
  -c "SELECT count(*) FROM budget_chunks;"
```

> Para este ejercicio basta con `budget_chunks`: la generación se fundamenta en el corpus de
> presupuestos. (Las colecciones `transcript_chunks` / `technical_doc_chunks` de la S10 no
> intervienen aquí.)

### 2.3 El mismo pipeline por HTTP (no solo en scripts)

El flujo online está expuesto en la API; los scripts de evaluación reutilizan **las mismas
funciones puras**. Equivalencias útiles:

| Paso del pipeline | Endpoint HTTP (producción) | Endpoint por etapa (didáctico) |
| --- | --- | --- |
| Pipeline completo transcript→estimate | `POST /v1/estimate/from-transcript` (auth `ESTIMATE_API_KEY`, 10/min, idempotente) | — |
| (1) reformular | — | `POST /v1/estimate/stages/reformulate` |
| (3) recuperar | `POST /v1/retrieval/search` (auth `RETRIEVAL_API_KEY`) | `POST /v1/estimate/stages/retrieve` |
| (4) ensamblar | — | `POST /v1/estimate/stages/assemble` |
| (5) generar | — | `POST /v1/estimate/stages/generate` |

Ejemplo del camino completo por HTTP:

```bash
curl -s -X POST http://localhost:8000/v1/estimate/from-transcript \
  -H "Content-Type: application/json" -H "X-API-Key: $ESTIMATE_API_KEY" \
  -d '{"transcript": "<transcripción de la reunión, ≥100 caracteres>"}'
```

La respuesta es el `Estimate` con las fuentes por línea; el contrato HTTP no cambia respecto
a la S9, solo se **enriquece** el cuerpo con la citación por línea.

---

## 3. Parte 1 — Citación verificable a nivel de línea

### 3.1 Schema extendido (Pydantic v2)

La citación se baja del nivel global (`SourceCitation`, S9) al nivel de **línea**. Cada tarea
(`TaskItem`) es una línea de estimación y transporta sus fuentes
([app/generation/rag/schemas.py](estimator/app/generation/rag/schemas.py)):

```python
class SourceReference(BaseModel):       # cita verificable a nivel de línea
    chunk_id: str       # id del <source> recuperado que respalda la línea
    document_id: str    # presupuesto histórico al que pertenece el chunk
    evidence: str       # fragmento/cifra VERBATIM de la fuente (no parafrasear)

class TaskItem(BaseModel):              # una línea de estimación
    name: str
    engineer_days: int | None = None
    grounded: bool = False              # False => sin datos suficientes
    sources: list[SourceReference] = []  # no vacío sii grounded=True
```

**Regla de integridad** (implementada como `@model_validator` en `TaskItem._grounding_integrity`):
una línea `grounded=True` debe citar ≥1 fuente; una línea `grounded=False` no puede llevar
fuentes **ni inventar horas** (`engineer_days` debe ser `None`). Si se viola, el validador
lanza y **Instructor reprompta** al modelo en vez de dejar pasar una línea no verificable.

### 3.2 Prompt de atribución por línea

El system prompt ([prompt_builder.py::build_system_prompt](estimator/app/generation/rag/prompt_builder.py)) obliga al modelo a:

1. Basar cada número **solo** en los bloques `<source>` (nada de conocimiento externo).
2. Por cada tarea derivada de evidencia: `grounded=true` + `sources` con `chunk_id` (el
   atributo `id` exacto del `<source>`), `document_id` (su atributo exacto) y `evidence`
   (un fragmento **verbatim** copiado, p. ej. nombre de componente + horas — no paráfrasis).
3. Citar **solo** ids que aparezcan literalmente en el bloque `<sources>`; nunca inventarlos.
4. Si una tarea no tiene soporte suficiente: `grounded=false`, `sources` vacío y
   `engineer_days` null — capturarla como `Assumption`, no estimar a ojo.

Los chunks llegan al prompt ya identificados: el ensamblador
([context_assembler.py](estimator/app/generation/rag/context_assembler.py)) envuelve cada uno
en `<source id="{chunk.id}" document_id="{document_id}" …>…</source>`, así que el `id` que el
modelo cita es exactamente el que luego se verifica.

### 3.3 Verificación post-generación

`verify_citations()` ([validation.py](estimator/app/generation/rag/validation.py)) recorre
todas las líneas (`modules[].tasks[]`) y comprueba que cada `chunk_id` citado esté en el
conjunto de ids realmente entregados al LLM. Firma:

```python
def verify_citations(estimate: Estimate, retrieved_chunk_ids: set[str]) -> CitationReport:
    """Flag any line whose cited chunk_id was never in the retrieved context."""
```

El `CitationReport` distingue tres estados por línea:

- **grounded** — todas las citas de la línea estaban en el contexto.
- **dangling** — al menos un `chunk_id` citado nunca se recuperó (alucinación de fuente).
- **insufficient** — la línea se marcó como sin datos suficientes (`grounded=False`).

También verifica las citaciones globales (`Estimate.sources`, capa S9), para que un id
inventado no pueda esconderse en ningún nivel. El resultado se loguea con `structlog`
correlacionado por `request_id`. Una citación colgante es un **fallo de calidad**, no
cosmético: el informe la deja visible (`dangling_citations`).

### 3.4 Prueba de la verificación (criterios de aceptación #1–#3)

[scripts/demo_verify_citations_s11.py](estimator/scripts/demo_verify_citations_s11.py)
construye una estimación con **una citación colgante plantada a propósito** (`chunk_id="999"`,
nunca recuperado) y una línea sin datos, y comprueba que la verificación la detecta:

```bash
uv run python scripts/demo_verify_citations_s11.py
```

Salida (real):

```
=== Citation verification report ===
lines: 4  grounded: 2  dangling: 1  insufficient: 1
verified citations: 2
dangling citations: ['999']

module                   line                     status        cited -> dangling
----------------------------------------------------------------------------------
Authentication & SCA     OAuth 2.0 backend        grounded      ['101'] -> -
PSD2 & Open Banking      Open banking connectors  grounded      ['102'] -> -
Ledger                   Transaction ledger       dangling      ['999'] -> ['999']
Reporting                Regulatory reporting     insufficient  [] -> -

ACCEPTANCE: PASS
```

Esto cubre los criterios #1 (cada línea grounded cita ≥1 fuente real), #2 (detecta la
colgante introducida a propósito) y #3 (las líneas sin soporte se marcan, no se rellenan).

### 3.5 Informe de citaciones sobre una estimación REAL (Q1)

Ejecutando el pipeline real sobre Q1 (banca móvil) y verificando contra los 5 chunks
recuperados:

```
Q1 — total_lines=34  grounded=21  dangling=0  insufficient=13  verified_citations=25  dangling=[]
```

Primeras líneas del informe (todas resuelven a chunks reales, **0 colgantes**):

| módulo | línea | estado | cita |
| --- | --- | --- | --- |
| Authentication & Access | Implement OAuth 2.0 authorization code/refresh flows | grounded | `['1']` |
| Authentication & Access | JWT-based session management | grounded | `['1']` |
| Authentication & Access | Multi-tenant token isolation | grounded | `['1']` |
| … (13 líneas marcadas `insufficient`: scope sin análogo histórico) | | insufficient | `[]` |

Lectura: sobre datos reales el generador **no alucina fuentes** (0 dangling en las 5
consultas) y marca honestamente como `insufficient` el scope sin soporte (13 líneas en Q1).
La citación verificable cumple su función.

---

## 4. Parte 2 — Evaluación RAGAS

### 4.1 Golden set extendido

`evals/golden_generation_s11.json` parte de las **5 consultas de la Sesión 10** (Q1–Q5,
budget-only; texto idéntico) y añade a cada una un `ground_truth`: la estimación de
referencia experta (engineer-days y módulos esperados). No se parte de cero: se **enriquece**
el set existente. (Las consultas cross-collection Q6–Q8 viven en `golden_retrieval.json` y
pertenecen al ejercicio de **retrieval** de la S10, no a este.)

### 4.2 Las cuatro métricas

| Métrica | Qué mide | Necesita |
| --- | --- | --- |
| **faithfulness** | Fracción de afirmaciones del `answer` que se infieren de los `contexts`. Baja = alucinación / saltos no respaldados. | LLM-juez |
| **answer_relevancy** | Cómo de relevante es el `answer` a la `question` (genera preguntas a partir del answer y mide similitud con la original). | LLM-juez + embeddings |
| **context_precision** | ¿Está la señal relevante bien rankeada en `contexts`? (relación señal/ruido del contexto recuperado). | LLM-juez |
| **context_recall** | Fracción de afirmaciones del `ground_truth` atribuibles a los `contexts`. Bajo = falta contexto para cubrir la referencia. | LLM-juez |

Juez: `gpt-4o-mini` (configurable en el golden set). Embeddings: `text-embedding-3-small`.

### 4.3 Cómo se ejecuta (dos pasos, por un conflicto de dependencias)

`ragas 0.4.x` importa incondicionalmente una ruta de Vertex (`langchain_community.chat_models.vertexai`)
que el `langchain-community` actual ya no expone. Por eso el flujo se parte en dos:

```bash
# Paso 1 — RECOLECTAR: corre el pipeline real (sin ragas) y vuelca las muestras.
#   Requiere DATABASE_URL al puerto del estimator-postgres y un timeout amplio
#   (gpt-5 con reasoning=high supera el LLM_TIMEOUT=30 por defecto).
LLM_TIMEOUT=600 \
DATABASE_URL="postgresql+psycopg://estimator:estimator@localhost:5433/estimator" \
  uv run python scripts/eval_ragas_s11.py --collect-only /tmp/ragas_samples.json

# Paso 2 — PUNTUAR: ragas + LLM-juez sobre las muestras ya recolectadas.
#   score_ragas_s11.py instala stubs para el import de Vertex (_install_vertex_shims)
#   y solo importa ragas/langchain_openai — NO la app. Necesita OPENAI_API_KEY en el entorno.
set -a; source <(grep -E '^OPENAI_API_KEY=' .env); set +a
uv run python scripts/score_ragas_s11.py /tmp/ragas_samples.json --out evals/ragas_baseline_s11_pre_exercise.json
```

> **Atajo en un paso** (`uv run python scripts/eval_ragas_s11.py`) — recolecta y puntúa en el
> mismo proceso, pero `run_ragas` importa ragas **sin** el shim de Vertex → fallaría con el
> `ModuleNotFoundError`. El flujo de dos pasos lo evita y, de paso, te deja las muestras
> cacheadas para re-puntuar sin volver a pagar la generación.

**Prerequisitos** (todos cumplidos en esta corrida): stack arriba, `budget_chunks` poblada
(60), `OPENAI_API_KEY` en `.env`, golden set con `ground_truth`, `uv sync` (trae ragas+datasets).

### 4.4 Tabla de métricas (baseline de calidad de generación)

Resultado real (`evals/ragas_baseline_s11_pre_exercise.json`), juez `gpt-4o-mini`, embeddings
`text-embedding-3-small`:

| query | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Q1 | 0.500 | 0.000 | 1.000 | 0.000 |
| Q2 | 0.364 | 0.079 | 1.000 | 0.000 |
| Q3 | 0.350 | 0.160 | 1.000 | 0.714 |
| Q4 | 0.588 | 0.000 | 1.000 | 0.667 |
| Q5 | 0.450 | 0.000 | 1.000 | 0.714 |
| **promedio** | **0.450** | **0.048** | **1.000** | **0.419** |

---

## 5. Nota — lo que más chirría de estos números

- **`total_engineer_days = 480` en Q1 = las 480 *horas* del presupuesto histórico
  BUD-2024-001.** El modelo copió la cifra **confundiendo horas con engineer-days**; la
  referencia experta son ~60 engineer-days → **8× de sobreestimación**. Es el fallo más
  grave y explica buena parte de la baja `faithfulness`/`answer_relevancy`: el corpus da
  horas y el schema pide días, y nadie convierte. (Coherente con la decisión de S10 de mover
  las horas a una búsqueda por-tarea posterior, no inferirlas aquí.)
- **`answer_relevancy ≈ 0.048` (casi cero en todas).** Muy probablemente un **artefacto de
  formato**: el `answer` es la estimación renderizada como texto estructurado
  (`Total: …`, módulos, bullets), no prosa tipo QA; RAGAS genera preguntas hipotéticas a
  partir del answer y casan mal con la pregunta original. Además salieron warnings
  `LLM returned 1 generations instead of requested 3` (el juez no devolvió las 3 generaciones
  que la métrica pide), lo que degrada su fiabilidad. **A interpretar con cautela, no como
  "la respuesta es irrelevante".**
- **`context_recall = 0.000` en Q1 y Q2** pese a `context_precision = 1.000`: lo recuperado
  es preciso, pero el `ground_truth` no se atribuye a los 5 chunks (recall bajo). Patrón
  "precisión perfecta, recall flojo": el top-k es correcto pero estrecho para cubrir toda la
  referencia experta.
- **`faithfulness ≈ 0.45`**: la mitad de las afirmaciones no se respaldan claramente en el
  contexto — coherente con el salto horas→días y con que el modelo decompone un componente
  histórico en muchas sub-tareas cuyo número exacto no está literalmente en la fuente.
- **`context_precision = 1.000` constante**: los 5 chunks recuperados son todos relevantes
  (corpus pequeño y bien separado, como vimos en la S10). El retrieval no es el cuello de
  botella; **la generación sí**.

---

## 6. Resumen para el directo

Lo mínimo que llevar a la sesión en vivo, en una pantalla.

**Baseline RAGAS (promedio sobre Q1–Q5):**

| faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|
| 0.450 | 0.048 | 1.000 | 0.419 |

**Verificación de citaciones sobre las 5 estimaciones reales del pipeline:**

| query | líneas | grounded | dangling | insufficient | citas verificadas | total (days) |
|---|---|---|---|---|---|---|
| Q1 | 34 | 21 | 0 | 13 | 25 | 480 |
| Q2 | 30 | 30 | 0 | 0 | 34 | 460 |
| Q3 | 23 | 23 | 0 | 0 | 27 | 71 |
| Q4 | 32 | 20 | 0 | 12 | 23 | 500 |
| Q5 | 40 | 40 | 0 | 0 | 45 | 1120 |
| **total** | **159** | **134** | **0** | **25** | **154** | |

**Titulares:**
- **Citaciones colgantes: 0/159 líneas.** La citación verificable funciona: cada línea
  `grounded` apunta a un chunk real; las 25 sin soporte se marcan `insufficient` (no inventan
  horas).
- **`context_precision` perfecto (1.0), `context_recall` flojo (0.42).** El retrieval trae lo
  relevante pero estrecho; el cuello de botella es la generación, no la recuperación.
- **`answer_relevancy` ≈ 0**: artefacto de formato (estimación estructurada vs métrica QA) +
  el juez devolviendo 1 generación en vez de 3. No leer como "respuesta irrelevante".
- **Bug de unidades:** 4/5 totales copian las **horas** históricas como **engineer-days**
  (Q1 480, Q5 1120…); solo Q3 (71) parece días reales. Es lo primero a atacar en el directo.

---

## 7. Comparación con la solución de referencia

Contraste de este run con el baseline de la solución
([evals/RAGAS_BASELINE_S11.md](estimator/evals/RAGAS_BASELINE_S11.md)). Mismo código, mismo
golden set, misma config (juez `gpt-4o-mini`, embeddings `text-embedding-3-small`) — pero
**dos ejecuciones distintas**, así que la comparación mide sobre todo la **reproducibilidad**.

### Ficheros de salida y su procedencia

Los dos datasets viven en ficheros separados para poder comparar sin tocar la referencia:

| Fichero | Contenido | Cómo se generó |
| --- | --- | --- |
| `evals/ragas_baseline_s11.json` | **Baseline de la solución** (intacto): avg `0.552 / 0.033 / 1.000 / 0.114` | Producido por la solución con el flujo de dos pasos (`eval_ragas_s11.py --collect-only` → `score_ragas_s11.py --out`). Es el fichero que referencia `RAGAS_BASELINE_S11.md`; se mantiene sin modificar. |
| `evals/ragas_baseline_s11_pre_exercise.json` | **Mi run local**: avg `0.450 / 0.048 / 1.000 / 0.419` | Generado en este análisis (§4.3 / §9): el mismo flujo de dos pasos, escribiendo en este fichero para no pisar el baseline de la solución. |

**RAGAS (promedios):**

| métrica | este run | solución | Δ |
|---|---|---|---|
| faithfulness | 0.450 | 0.552 | −0.102 |
| answer_relevancy | 0.048 | 0.033 | +0.015 |
| context_precision | 1.000 | 1.000 | 0.000 |
| context_recall | 0.419 | 0.114 | +0.305 |

**Verificación de citaciones (totales):**

| | este run | solución |
|---|---|---|
| líneas | 159 | 169 |
| grounded | 134 | 116 |
| **dangling** | **0** | **0** |
| insufficient | 25 | 53 |
| citas verificadas | 154 | 144 |

### Qué es estable y qué no

**Conclusiones reproducibles (idénticas en ambos runs) — son las fiables:**
- **`context_precision = 1.000`** exacto en las dos corridas, en las 5 consultas. El corpus es
  pequeño y bien separado; el retrieval no es el problema.
- **`dangling = 0`** en ambas (0/159 y 0/169). La citación verificable es robusta: **nunca**
  apareció una fuente alucinada. Es el criterio de aceptación clave y se cumple
  independientemente del run.
- **`answer_relevancy` ≈ 0** en ambas (0.048 vs 0.033). El artefacto de formato es sistemático,
  no ruido de un run.
- **Mismos hallazgos cualitativos**: bug horas→days, precisión-perfecta/recall-flojo,
  `faithfulness` mediocre pese a citación real. Las dos notas llegan a las mismas conclusiones.

**Lo que varía entre runs (no fiarse del número exacto):**
- **`context_recall`** es lo más volátil: 0.419 aquí vs 0.114 en la solución. La causa es la
  **no-determinación del pipeline**: `reformulate_query` (gpt-5-mini) produce un search_text
  distinto en cada run → se recuperan chunks ligeramente distintos → el juez atribuye el
  `ground_truth` al contexto de forma distinta. Mi run recuperó contextos que cubren mejor la
  referencia en Q3/Q4/Q5 (0.714/0.667/0.714) frente a la solución, donde casi todo cae a 0.
- **`faithfulness`** se mueve por consulta (p. ej. Q4: 0.588 aquí vs 0.877 en la solución) por
  la estocasticidad del generador (gpt-5, reasoning) y del juez. Promedios en el mismo orden
  de magnitud (~0.45–0.55).
- **Reparto grounded/insufficient y los totales en days**: difieren por las distintas
  generaciones. Ejemplo del bug de unidades: aquí el "caso correcto" es Q3 (71 days); en la
  solución es Q4 (63 days) — distinta consulta acierta las unidades en cada run, lo que prueba
  que es inconsistencia del modelo, no una propiedad de una consulta concreta.

### Conclusión de la comparación

El baseline es **direccional, no una cifra exacta**: lo que se reproduce son los *diagnósticos*
(0 colgantes, precision 1.0, recall lastrado, answer_relevancy artefactual, bug de unidades),
no los valores puntuales. Para un baseline estable habría que (a) fijar/relajar la
no-determinación (temperatura, reformulación determinista o cacheada) y (b) promediar varias
corridas. Para el objetivo del ejercicio —traer un baseline y los hallazgos al directo— ambos
runs coinciden en lo que importa.

---

## 8. Criterios de aceptación

| # | Criterio | Estado |
| --- | --- | --- |
| 1 | Cada línea `grounded=True` cita ≥1 fuente real del contexto | ✅ demo + Q1 real (0 dangling) |
| 2 | La verificación detecta una citación colgante introducida a propósito | ✅ demo (`dangling=['999']`) |
| 3 | Las líneas sin soporte se marcan `insufficient`, no se rellenan | ✅ demo + Q1 (13 insufficient) |
| 4 | RAGAS devuelve las 4 métricas para las 5 consultas + promedio | ✅ tabla §4.4 |

---

## 9. Anexo — reproducción completa

```bash
# 0) Stack + datos
docker compose up -d
docker compose run --rm estimator python scripts/query_examples.py        # budgets → budget_chunks (60)

# 1) Verificación de citaciones (offline, sin red/DB) — criterios #1–#3
uv run python scripts/demo_verify_citations_s11.py

# 2) RAGAS — paso recolectar (pipeline real; gpt-5 reasoning alto → timeout amplio)
LLM_TIMEOUT=600 \
DATABASE_URL="postgresql+psycopg://estimator:estimator@localhost:5433/estimator" \
  uv run python scripts/eval_ragas_s11.py --collect-only /tmp/ragas_samples.json

# 3) RAGAS — paso puntuar (juez + embeddings; necesita OPENAI_API_KEY en el entorno)
set -a; source <(grep -E '^OPENAI_API_KEY=' .env); set +a
uv run python scripts/score_ragas_s11.py /tmp/ragas_samples.json --out evals/ragas_baseline_s11_pre_exercise.json
```

Salidas: `evals/ragas_baseline_s11_pre_exercise.json` (tabla + per-query + `citation_report`
por consulta) y, por pantalla, la tabla de métricas y el informe de verificación de
citaciones. (El baseline de la solución, `evals/ragas_baseline_s11.json`, **no se toca**.)

> **Notas de entorno descubiertas en esta corrida** (no son bugs de la solución):
> (1) `LLM_TIMEOUT=30` se queda corto para `gpt-5` con `reasoning_effort=high` → usar 600;
> (2) `score_ragas_s11.py` no carga `.env`, hay que exportar `OPENAI_API_KEY` al entorno.
