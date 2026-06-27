# Session 10 — Búsqueda híbrida y reranking: mediciones y respuestas

> Alcance del ejercicio: **únicamente** búsqueda híbrida (full-text + RRF) y reranking
> recall-then-rerank. No se evalúan expansión de consultas, routing multi-índice ni
> filtrado por metadatos (se construyen en la sesión en vivo). El código del proyecto ya
> implementa los pasos 1–3; este documento recoge el **análisis de medición (paso 4)** y
> las **conclusiones (paso 5)** que faltaban.

## Reproducibilidad

### Prerequisitos

La medición necesita tres cosas: el stack arriba, una API key de OpenAI (embeddings +
descarga del cross-encoder) y el corpus de presupuestos ingestado.

```bash
# 1) Stack arriba (FastAPI + Redis + estimator-postgres con pgvector)
docker compose up -d

# 2) API key presente (en .env)
grep ^OPENAI_API_KEY= .env
```

#### Ingesta de los budgets — de dónde se cargan y comando exacto

Los presupuestos se cargan desde **`data/budgets_sample.json`** (17 presupuestos, cada uno
con sus componentes). La ingesta es **por HTTP**: el script `scripts/query_examples.py`
recorre el fichero y hace **un `POST /embeddings/ingest` por presupuesto** (un documento por
request — `scripts/query_examples.py:91-97`):

```python
for budget in budgets:
    client.post(f"{base_url}/embeddings/ingest", json={
        "source_path": f"data/budgets_sample.json::{budget['budget_id']}",
        "document_type": "historical_budget",
        "content": budget,
    })
```

Cada presupuesto se trocea por componente (chunker estructural) y se persiste en la tabla
**`budget_chunks`** con su embedding (pgvector) y su `content_tsv` (full-text). Comando
exacto que se usó:

```bash
docker compose run --rm estimator python scripts/query_examples.py
```

> Es idempotente: los documentos ya persistidos responden `409` y se omiten, así que
> re-ejecutarlo no duplica datos. (La API debe estar arriba, porque el script habla por HTTP;
> de ahí el `docker compose up -d` previo.)

#### Verificación de prerequisitos

Antes de medir, comprobar que el corpus está realmente en la BD (resultado esperado entre
paréntesis):

```bash
# Presupuestos ingestados        → esperado: 17
docker compose exec -T estimator-postgres psql -U estimator -d estimator -t \
  -c "SELECT count(*) FROM documents WHERE document_type='historical_budget';"

# Chunks de presupuesto          → esperado: 60
docker compose exec -T estimator-postgres psql -U estimator -d estimator -t \
  -c "SELECT count(*) FROM budget_chunks;"

# La columna full-text está poblada (hybrid)  → esperado: 60 (= sin NULLs)
docker compose exec -T estimator-postgres psql -U estimator -d estimator -t \
  -c "SELECT count(*) FROM budget_chunks WHERE content_tsv IS NOT NULL;"
```

Si `budget_chunks` da `0`, los budgets no están ingestados → ejecutar el `query_examples.py`
de arriba antes de medir.

#### Pre-flight del reranker (gate del ejercicio)

Antes de medir las configuraciones con reranking (C y D), conviene confirmar que el
cross-encoder se descarga, carga y puntúa correctamente. El módulo
`app/generation/rag/retrieval/verify_reranker.py` hace ese chequeo de cordura: carga el
modelo configurado y puntúa un par de prueba en el que un documento es obviamente más
relevante que el otro, exigiendo que el relevante puntúe más alto.

```bash
# OJO: en este repo el servicio se llama `estimator` (el enunciado pone `ai-service`).
docker compose exec estimator python -m app.generation.rag.retrieval.verify_reranker
```

Salida esperada (exit code `0` = entorno listo; `1` = no carga/ejecuta; `2` = carga pero
rankea mal — revisar el nombre del modelo):

```
Loading reranker model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 ...
Query:        'e-commerce checkout and shopping cart platform'
Relevant doc:   score = 2.9526
Irrelevant doc: score = -9.0109
OK: reranker loaded and ranked the relevant document first.
```

### Ejecución de la medición

Las cuatro configuraciones se invocan con un único harness enfocado, que reutiliza las
mismas funciones puras del pipeline y el mismo criterio de relevancia que el harness
multi-índice (`scripts/eval_retrieval_s10.py`), pero limitado al alcance del ejercicio
(A–D sobre las 5 consultas budget-only):

```bash
# El harness lee evals/ (no montado en el contenedor) → se corre en el host.
# DATABASE_URL apunta al puerto del estimator-postgres publicado en el host (5433).
export DATABASE_URL="postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
uv run python scripts/eval_s10_hybrid_rerank.py
```

- **Corpus**: `data/budgets_sample.json` → 60 chunks en `budget_chunks` (un chunk por
  componente de presupuesto).
- **Método**: por cada (consulta, configuración) se descarta una ejecución de warm-up y se
  miden 3 ejecuciones. La latencia es **extremo a extremo** del retrieval (embedding de la
  consulta + ramas vectorial/léxica + fusión + rerank). Se usa un umbral de distancia
  permisivo (`2.0`) para que el top-k nunca se trunque por el filtro de relevancia: se
  compara **calidad de ranking**, no la puerta de soft-fail.
- **Métrica**: `precision@5` = (chunks del top-5 cuyo presupuesto de origen está en el
  conjunto relevante anotado) / 5. Como un presupuesto produce varios chunks, el top-5
  puede saturar a 1.00 aun con pocos presupuestos relevantes.

---

## Paso 1–3 — Resumen de la implementación (ya existente)

| Paso | Qué hace | Dónde |
| --- | --- | --- |
| 1. Full-text en PostgreSQL | Columna `content_tsv tsvector GENERATED ALWAYS … STORED` sobre `content`, con índice **GIN**. | `alembic/versions/0003_session10_fts.py` |
| 2. Rama léxica + fusión RRF | Búsqueda por palabras clave (`ts_rank_cd`) fusionada con la vectorial mediante Reciprocal Rank Fusion; ranking único. Constante de suavizado `RRF_K` (default **60**). | `app/generation/rag/retrieval/fusion.py` (`reciprocal_rank_fusion`), `pipeline.py` (`fused_order`) |
| 3. Reranker (recall-then-rerank) | Cross-encoder reordena un pool amplio (recall top-50) y se queda con el top-5. Activable/desactivable sin tocar código. | `app/generation/rag/retrieval/reranker.py`; toggle `RERANKER_ENABLED` (.env) y en runtime vía `PUT /api/v1/config/retrieval` |

El modo de búsqueda se expone como toggle (`search_mode` = `vector` / `hybrid`) y el rerank
como flag (`rerank` on/off), ambos parametrizables por configuración — es lo que el harness
recorre de forma reproducible.

> **Nota sobre el idioma de la text-search config.** El enunciado asume presupuestos en
> español. El corpus de muestra que se distribuye con el proyecto (`budgets_sample.json`)
> está **en inglés**, por lo que la migración 0003 usa deliberadamente la configuración
> `english` (stemming/stop-words coherentes con el texto real). Cambiar a un corpus en
> español solo requiere sustituir el `regconfig` (`'english'` → `'spanish'`) en la
> migración; la mecánica de búsqueda híbrida es idéntica.

---

## Paso 4 — Golden set y medición

### Para qué sirve el golden set y cómo se usa en la comparación

El golden set es el **patrón de verdad** ("ground truth") contra el que se mide si la
búsqueda recupera lo correcto. Sin él solo verías listas de resultados sin saber cuáles
están bien; con él, "esto parece funcionar" se convierte en un **número objetivo**.

Es un conjunto de consultas representativas del dominio, cada una con la respuesta correcta
**anotada a mano**: qué presupuestos son *de verdad* relevantes para esa consulta. Un humano
experto decide, por ejemplo, que para Q1 los relevantes son `BUD-2024-001` y `BUD-2024-003`.
Es subjetivo pero **fijo**: es la vara de medir.

A partir de ahí, la métrica de calidad es la **precisión@5**: por cada consulta se miran los
5 primeros resultados y se cuenta cuántos están en la lista de relevantes anotados, luego se
promedia sobre las 5 consultas.

```
precision@5 = (resultados del top-5 que SÍ están en relevant_budget_ids) / 5
```

**El punto clave de la comparación**: el golden set se mantiene **idéntico** mientras se
cambia **solo la configuración de búsqueda**. Es un experimento controlado — mismas
preguntas, misma verdad, distinta máquina de buscar:

```
                  ┌─────────────────────────────────────────┐
   Golden set ──▶ │  A: Vectorial,  sin rerank               │ ──▶ precision@5, latencia
   (5 consultas + │  B: Híbrida,    sin rerank               │ ──▶ precision@5, latencia
    verdad fija)  │  C: Vectorial,  con rerank               │ ──▶ precision@5, latencia
                  │  D: Híbrida,    con rerank               │ ──▶ precision@5, latencia
                  └─────────────────────────────────────────┘
                         ↑ lo único que cambia entre filas
```

Como la única variable que cambia entre filas es la configuración, **cualquier diferencia en
precisión@5 es atribuible a la configuración**, no al azar de qué se preguntó. Eso es lo que
permite afirmar con datos cosas como "la híbrida no mejora a la vectorial aquí" o "el
reranking no sube la precisión pero multiplica la latencia ×35", y responder al paso 5 con
evidencia en lugar de intuición.

**Ejemplo concreto (Q1).** Verdad anotada = `{BUD-2024-001, BUD-2024-003}`. Supón que una
configuración devuelve este top-5 (a nivel de *chunk* — cada presupuesto aporta varios):

| # | chunk recuperado | ¿su presupuesto está en la verdad? |
|---|---|---|
| 1 | BUD-2024-001 / auth | ✅ |
| 2 | BUD-2024-001 / ledger | ✅ |
| 3 | BUD-2024-003 / pagos | ✅ |
| 4 | BUD-2024-006 / payouts e-commerce | ❌ (el *distractor* trampa) |
| 5 | BUD-2024-001 / API | ✅ |

→ 4 aciertos de 5 → **precision@5 = 0.80**.

El **distractor** (BUD-2024-006, "parece pagos" pero es e-commerce) está puesto a propósito
en las anotaciones: es la trampa que separa una búsqueda buena de una que se deja engañar por
vocabulario parecido. Si una configuración lo cuela en el top-5 y otra lo evita, ahí es donde
la comparación se vuelve informativa.

### Golden set (5 consultas, anotadas a mano)

Definido en `evals/golden_retrieval.json` (Q1–Q5; Q6–Q8 son cross-collection y quedan
fuera de alcance). Cada consulta es una descripción de proyecto a estimar, con los
presupuestos realmente relevantes anotados y un *distractor* deliberado (mismo sector o
vocabulario cercano pero funcionalidad distinta) para tensar el ranking.

| ID | Consulta (resumen) | Presupuestos relevantes | Distractor |
| --- | --- | --- | --- |
| Q1 | Banca móvil: OAuth2, PSD2/SCA, ledger de transacciones | BUD-2024-001, 003 | BUD-2024-006 (payouts e-commerce, "parece pagos") |
| Q2 | E-commerce headless: catálogo, checkout, recomendaciones | BUD-2024-005, 006, 007, 017 | pagos finance / scheduling healthcare |
| Q3 | Telemedicina: citas, vídeo-consulta, HL7/FHIR, historia clínica | BUD-2024-009, 010 | BUD-2024-011/012 (mismo sector salud, otra función) |
| Q4 | IoT industrial: telemetría de sensores, mantenimiento predictivo | BUD-2024-013, 015 | BUD-2024-014 (AGV logística, no telemetría) |
| Q5 | Pasarela de pagos real-time: fraude, conciliación, doble entrada | BUD-2024-003, 001 | BUD-2024-006/007 (léxicamente cercanos, dominio erróneo) |

### Configuraciones

| Config | Búsqueda | Reranking |
| --- | --- | --- |
| A | Vectorial | No |
| B | Híbrida | No |
| C | Vectorial | Sí |
| D | Híbrida | Sí |

### Tabla comparativa (precisión@5 y latencia)

| Config | Búsqueda | Reranking | Precision@5 | Latencia (ms) |
| --- | --- | --- | --- | --- |
| A | Vectorial | No  | **0.92** | **223.3** |
| B | Híbrida   | No  | **0.92** | **208.4** |
| C | Vectorial | Sí  | **0.92** | 7730.9 |
| D | Híbrida   | Sí  | **0.92** | 8054.1 |

### Desglose precision@5 por consulta

| Query | A | B | C | D |
| --- | --- | --- | --- | --- |
| Q1 | 1.00 | 1.00 | 0.80 | 0.80 |
| Q2 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q3 | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4 | 0.80 | 0.80 | 1.00 | 1.00 |
| Q5 | 1.00 | 1.00 | 1.00 | 1.00 |

### Lectura de los resultados

- **Vectorial ≈ Híbrida** en este corpus: A y B son idénticas en precisión (0.92) y casi
  iguales en latencia (~210–223 ms). La rama vectorial ya satura el top-5 con chunks del
  presupuesto correcto, así que la rama léxica fusionada por RRF no añade aciertos netos
  (tampoco los rompe). Tiene sentido: las consultas son descripciones semánticas, no
  identificadores literales — el escenario donde lo léxico brilla (SKUs, "IDoc/BAPI", IDs)
  está en Q6–Q8, fuera de alcance.
- **El reranking no mueve la precisión@5 media** (0.92 → 0.92). A nivel de consulta solo
  **reordena dentro del conjunto relevante**: mejora Q4 (0.80 → 1.00) pero empeora Q1
  (1.00 → 0.80), compensándose. No rescata ningún relevante que vectorial/híbrida no
  trajeran ya en el top-5.
- **El reranking multiplica la latencia ~35×** (≈210 ms → ≈7.9 s). El coste es el scoring
  del cross-encoder sobre 50 pares (`reranker_scored … score_ms≈9–10 s` en los logs),
  dominado por CPU con el modelo multilingüe `mmarco-mMiniLMv2-L12-H384-v1`.

> **Cautelas metodológicas.** (1) Golden set pequeño (5 consultas). (2) La precisión es a
> nivel de *chunk* con origen relevante; como cada presupuesto aporta varios chunks, la
> métrica satura fácilmente y es poco sensible al reordenamiento fino — un `recall@k` a
> nivel de documento o un `MRR` discriminaría mejor el aporte del reranker. (3) La latencia
> del reranker es CPU-bound; con GPU real bajaría, pero seguiría siendo órdenes de magnitud
> sobre la búsqueda sin rerank.

### Por qué el reranking no aporta aquí: tamaño del corpus

La causa raíz de que el reranking no mueva la precisión es **el tamaño del corpus**, no un
fallo de implementación (los logs confirman que el cross-encoder puntúa: `reranker_scored …
pairs=50`). Estado actual de la BD:

| | Cantidad |
| --- | --- |
| Presupuestos (`documents`, `document_type='historical_budget'`) | **17** |
| Chunks en `budget_chunks` (componentes) | **60** |
| Presupuestos en `data/budgets_sample.json` | **17** |

**El corpus de presupuestos está completo**: los 17 presupuestos del fichero de muestra
están ingestados (sus 60 componentes = 60 chunks). No falta ninguno.

Con solo 60 chunks, el patrón recall-then-rerank recupera *top-50* antes de reordenar, así
que el reranker está **reordenando casi todo el corpus** en cada consulta. No existe un pool
amplio de candidatos "casi relevantes" donde el cross-encoder pueda marcar diferencia, y
como las 5 consultas son de dominios bien diferenciados (banca, e-commerce, salud, IoT,
pagos), la primera pasada vectorial ya devuelve un top-5 casi perfecto. El reranker no tiene
nada que rescatar — solo reordena dentro de lo ya acertado, de ahí el delta neto cero en
precisión@5.

> **Nota — datos adicionales sin ingestar.** `data/task_corpus.json` contiene 60 proyectos
> sintéticos (~1.5k tareas) con `budget_id`, pero **no es el corpus de presupuestos**: es el
> corpus a nivel de tarea (`chunk_type='historical_task'`) para la búsqueda de horas por
> tarea (`POST /v1/estimate/tasks/hours`), y se ingesta aparte con
> `scripts/build_task_corpus.py --ingest`. Aterriza en la misma tabla `budget_chunks` (solo
> cambia el `chunk_type`), por lo que ingestarlo añadiría ~1.5k distractores al eval A–D
> (que filtra por colección `budget` pero no por `chunk_type`) — y ahí el reranking sí
> tendría margen para mostrar valor. Pero eso mezcla dos corpus distintos y cambia el
> experimento del ejercicio, así que se deja fuera de esta medición.

---

## Paso 5 — Conclusiones

**¿Qué configuración usaría en el proyecto?** Para este corpus y este caso de uso,
**Configuración A (vectorial, sin reranking)**. Da la misma precisión@5 que las otras tres
(0.92) a ~210 ms, sin ninguna dependencia adicional (ni cross-encoder ni la rama léxica).
La búsqueda híbrida (B) es igual de buena y prácticamente gratis, así que la dejaría
**activada por configuración** como red de seguridad para consultas con identificadores
literales (el patrón que aparecerá en transcripciones y docs técnicos), pero sobre las
descripciones de presupuesto puro no aporta hoy.

**¿La ganancia de relevancia del reranking justifica su latencia aquí?** No. En este caso
concreto el reranking **no aporta ganancia neta de relevancia** (precisión@5 media idéntica;
solo reordena dentro de lo ya recuperado) y a cambio multiplica la latencia por ~35,
pasando de ~0.2 s a ~8 s por consulta — inviable para un flujo interactivo de estimación.
La conclusión es específica de este escenario: el corpus es pequeño y semánticamente
separable, así que la recuperación de primera pasada ya es casi perfecta y no deja margen
al reordenador. En un corpus mayor y más ruidoso, con muchos candidatos "casi relevantes"
en el pool de recall y una métrica sensible al orden (MRR / nDCG / recall@k a nivel de
documento), el reranking sí podría justificar su coste — y por eso se mantiene como toggle
desactivable, no como código fijo.
