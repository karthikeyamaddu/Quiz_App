import React, { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000'

function App() {
  const [pdfFile, setPdfFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [questionMode, setQuestionMode] = useState('exact')
  const [questionCount, setQuestionCount] = useState(10)
  const [finetuneQuestionCount, setFinetuneQuestionCount] = useState(2)
  const [ragQuestionCount, setRagQuestionCount] = useState(5)
  const [chunkSize, setChunkSize] = useState(800)
  const [chunkOverlap, setChunkOverlap] = useState(150)
  const [ollamaModel, setOllamaModel] = useState('llama2-uncensored')
  const [geminiPrimary, setGeminiPrimary] = useState('gemini-2.5-flash')
  const [geminiSecondary, setGeminiSecondary] = useState('gemini-flash-latest')

  const [extractStage, setExtractStage] = useState(null)
  const [chunkStage, setChunkStage] = useState(null)
  const [finetuneStage, setFinetuneStage] = useState(null)
  const [ragStage, setRagStage] = useState(null)
  const [primaryStage, setPrimaryStage] = useState(null)
  const [secondaryStage, setSecondaryStage] = useState(null)
  const [fullStage, setFullStage] = useState(null)
  const [expandedChunks, setExpandedChunks] = useState({})

  const canChunk = !!extractStage?.text
  const canFinetune = !!chunkStage?.chunks?.length
  const canRag = !!chunkStage?.chunks?.length && !!finetuneStage?.questions?.length
  const canPrimary = !!extractStage?.text
  const canSecondary = !!ragStage?.questions?.length || !!primaryStage?.questions?.length

  const effectiveQuestionCount = useMemo(() => {
    if (questionMode === 'auto') {
      const text = extractStage?.text || ''
      return Math.min(25, Math.max(3, Math.floor(text.split(/\s+/).filter(Boolean).length / 120)))
    }
    return questionCount
  }, [questionMode, questionCount, extractStage])

  const setApiError = async (response) => {
    try {
      const payload = await response.json()
      throw new Error(payload.error || 'Request failed')
    } catch (parseErr) {
      throw new Error(parseErr.message || 'Request failed')
    }
  }

  const resetPipeline = () => {
    setError('')
    setExtractStage(null)
    setChunkStage(null)
    setFinetuneStage(null)
    setRagStage(null)
    setPrimaryStage(null)
    setSecondaryStage(null)
    setFullStage(null)
    setExpandedChunks({})
  }

  const runFullPipeline = async () => {
    if (!pdfFile) {
      setError('Please upload a PDF first.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const form = new FormData()
      form.append('file', pdfFile)
      form.append('questionMode', questionMode)
      form.append('questionCount', String(questionCount))
      form.append('finetuneQuestionCount', String(finetuneQuestionCount))
      form.append('ragQuestionCount', String(ragQuestionCount))
      form.append('chunkSize', String(chunkSize))
      form.append('chunkOverlap', String(chunkOverlap))
      form.append('ollamaModel', ollamaModel)
      form.append('geminiModelPrimary', geminiPrimary)
      form.append('geminiModelSecondary', geminiSecondary)

      const response = await fetch(`${API_BASE}/api/pipeline/full`, {
        method: 'POST',
        body: form,
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()

      setFullStage(payload)
      setExtractStage({
        filename: payload.input?.filename,
        text: payload.extractedText,
        textLength: payload.input?.textLength,
        wordCount: payload.input?.wordCount,
      })
      setChunkStage({
        count: payload.chunks?.length || 0,
        chunkSize: payload.config?.chunkSize,
        chunkOverlap: payload.config?.chunkOverlap,
        chunks: payload.chunks || [],
      })
      setFinetuneStage(payload.finetune)
      setRagStage(payload.rag)
      setPrimaryStage(payload.primaryGemini)
      setSecondaryStage(payload.secondaryGemini)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runExtract = async () => {
    if (!pdfFile) {
      setError('Please upload a PDF first.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', pdfFile)
      const response = await fetch(`${API_BASE}/api/stage/extract`, {
        method: 'POST',
        body: form,
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setExtractStage(payload)
      setChunkStage(null)
      setFinetuneStage(null)
      setRagStage(null)
      setPrimaryStage(null)
      setSecondaryStage(null)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runChunk = async () => {
    if (!extractStage?.text) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/stage/chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: extractStage.text,
          chunkSize,
          chunkOverlap,
        }),
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setChunkStage(payload)
      setFinetuneStage(null)
      setRagStage(null)
      setSecondaryStage(null)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runFinetune = async () => {
    if (!chunkStage?.chunks?.length) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/stage/finetune`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunks: chunkStage.chunks,
          questionCount: finetuneQuestionCount,
        }),
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setFinetuneStage(payload)
      setRagStage(null)
      setSecondaryStage(null)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runRag = async () => {
    if (!canRag) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/stage/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunks: chunkStage.chunks,
          questions: finetuneStage.questions,
          ragQuestionCount,
          ollamaModel,
        }),
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setRagStage(payload)
      setSecondaryStage(null)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runPrimaryGemini = async () => {
    if (!extractStage?.text) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/stage/gemini-primary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: extractStage.text,
          questionCount: effectiveQuestionCount,
          geminiModelPrimary: geminiPrimary,
        }),
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setPrimaryStage(payload)
      setSecondaryStage(null)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runSecondaryGemini = async () => {
    if (!canSecondary) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/stage/gemini-secondary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ragQuestions: ragStage?.questions || [],
          primaryQuestions: primaryStage?.questions || [],
          questionCount: effectiveQuestionCount,
          geminiModelSecondary: geminiSecondary,
        }),
      })
      if (!response.ok) {
        await setApiError(response)
      }
      const payload = await response.json()
      setSecondaryStage(payload)
      setFullStage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const downloadDebugJson = () => {
    const payload = {
      config: {
        questionMode,
        questionCount,
        finetuneQuestionCount,
        ragQuestionCount,
        chunkSize,
        chunkOverlap,
        ollamaModel,
        geminiPrimary,
        geminiSecondary,
      },
      extractStage,
      chunkStage,
      finetuneStage,
      ragStage,
      primaryStage,
      secondaryStage,
      fullStage,
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `quiz_pipeline_debug_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const toggleChunk = (index) => {
    setExpandedChunks((prev) => ({ ...prev, [index]: !prev[index] }))
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Quiz Pipeline Debugger</h1>
        <p>PDF → Chunks → Finetuned Model → RAG Validation → Primary Gemini → Secondary Gemini</p>
      </header>

      <section className="panel">
        <h2>Input & Config</h2>
        <div className="grid">
          <label>
            PDF File
            <input
              type="file"
              accept="application/pdf"
              onChange={(event) => setPdfFile(event.target.files?.[0] || null)}
            />
          </label>

          <label>
            Question Mode
            <select value={questionMode} onChange={(event) => setQuestionMode(event.target.value)}>
              <option value="exact">Exact Count</option>
              <option value="auto">Auto From PDF Length</option>
            </select>
          </label>

          <label>
            Question Count
            <input
              type="number"
              min="1"
              max="50"
              value={questionCount}
              disabled={questionMode === 'auto'}
              onChange={(event) => setQuestionCount(Number(event.target.value) || 1)}
            />
          </label>

          <label>
            Finetune Questions
            <input
              type="number"
              min="1"
              max="50"
              value={finetuneQuestionCount}
              onChange={(event) => setFinetuneQuestionCount(Number(event.target.value) || 1)}
            />
          </label>

          <label>
            RAG Questions to Validate
            <input
              type="number"
              min="1"
              max="50"
              value={ragQuestionCount}
              onChange={(event) => setRagQuestionCount(Number(event.target.value) || 5)}
            />
          </label>

          <label>
            Chunk Size
            <input
              type="number"
              min="200"
              value={chunkSize}
              onChange={(event) => setChunkSize(Number(event.target.value) || 800)}
            />
          </label>

          <label>
            Chunk Overlap
            <input
              type="number"
              min="0"
              value={chunkOverlap}
              onChange={(event) => setChunkOverlap(Number(event.target.value) || 150)}
            />
          </label>

          <label>
            Ollama Model
            <input value={ollamaModel} onChange={(event) => setOllamaModel(event.target.value)} />
          </label>

          <label>
            Primary Gemini
            <input value={geminiPrimary} onChange={(event) => setGeminiPrimary(event.target.value)} />
          </label>

          <label>
            Secondary Gemini
            <input value={geminiSecondary} onChange={(event) => setGeminiSecondary(event.target.value)} />
          </label>
        </div>

        <div className="summary-row">
          <span>Effective question count: <strong>{effectiveQuestionCount}</strong></span>
          <span>Finetune question count: <strong>{finetuneQuestionCount}</strong></span>
          <span>RAG question count: <strong>{ragQuestionCount}</strong></span>
          <span>API: <strong>{API_BASE}</strong></span>
        </div>

        <div className="actions">
          <button disabled={loading || !pdfFile} onClick={runFullPipeline}>
            Run Full Pipeline
          </button>
          <button disabled={loading} onClick={resetPipeline}>
            Reset
          </button>
          <button
            disabled={loading || (!fullStage && !extractStage && !chunkStage && !finetuneStage && !ragStage && !primaryStage && !secondaryStage)}
            onClick={downloadDebugJson}
          >
            Download Debug JSON
          </button>
        </div>

        {loading && <p className="status">Running...</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="panel">
        <h2>Manual Stage Controls</h2>
        <div className="actions">
          <button disabled={loading || !pdfFile} onClick={runExtract}>1) Extract</button>
          <button disabled={loading || !canChunk} onClick={runChunk}>2) Chunk</button>
          <button disabled={loading || !canFinetune} onClick={runFinetune}>3) Finetune</button>
          <button disabled={loading || !canRag} onClick={runRag}>4) RAG Validate</button>
          <button disabled={loading || !canPrimary} onClick={runPrimaryGemini}>5) Primary Gemini</button>
          <button disabled={loading || !canSecondary} onClick={runSecondaryGemini}>6) Secondary Gemini</button>
        </div>
      </section>

      <StagePanel title="Extracted PDF Text" payload={extractStage}>
        {extractStage?.text && (
          <div className="text-box">{extractStage.text.slice(0, 2500)}{extractStage.text.length > 2500 ? ' ...' : ''}</div>
        )}
      </StagePanel>

      <StagePanel title="Chunks" payload={chunkStage}>
        {chunkStage?.chunks?.map((chunk) => {
          const expanded = !!expandedChunks[chunk.index]
          const preview = chunk.text.slice(0, 300)
          return (
            <div className="chunk-card" key={chunk.index}>
              <div className="chunk-head">
                <span>Chunk #{chunk.index} ({chunk.length} chars)</span>
                <button onClick={() => toggleChunk(chunk.index)}>{expanded ? 'Collapse' : 'Expand'}</button>
              </div>
              <p>{expanded ? chunk.text : `${preview}${chunk.text.length > 300 ? ' ...' : ''}`}</p>
            </div>
          )
        })}
      </StagePanel>

      <StagePanel title="Finetuned Model Output" payload={finetuneStage}>
        <QuestionList questions={finetuneStage?.questions} />
        <RawOutputList rawOutputs={finetuneStage?.rawOutputs} />
      </StagePanel>

      <StagePanel title="RAG Validation Output" payload={ragStage}>
        <QuestionList questions={ragStage?.questions} showRag />
      </StagePanel>

      <StagePanel title="Primary Gemini Output" payload={primaryStage}>
        <QuestionList questions={primaryStage?.questions} />
      </StagePanel>

      <StagePanel title="Secondary Gemini Output (Final)" payload={secondaryStage}>
        <h3>Final Quiz</h3>
        <QuestionList questions={secondaryStage?.finalQuiz} showScore />
      </StagePanel>
    </div>
  )
}

function StagePanel({ title, payload, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {!payload && <p className="muted">No data yet.</p>}
      {children}
      {payload && (
        <details>
          <summary>Raw JSON</summary>
          <pre>{JSON.stringify(payload, null, 2)}</pre>
        </details>
      )}
    </section>
  )
}

function QuestionList({ questions = [], showRag = false, showScore = false }) {
  if (!questions?.length) {
    return <p className="muted">No questions.</p>
  }

  return (
    <div className="question-list">
      {questions.map((item, index) => (
        <article className="question-card" key={`${item.question}-${index}`}>
          <p><strong>Q{index + 1}:</strong> {item.question}</p>
          <ol type="A">
            {(item.options || []).map((option, optIndex) => (
              <li key={`${option}-${optIndex}`}>{option}</li>
            ))}
          </ol>
          <p><strong>Correct:</strong> {item.correctAnswer}</p>
          <p><strong>Fallback Used:</strong> {item.usedFallback ? 'Yes' : 'No'}</p>
          {item.fallbackReason && <p><strong>Fallback Reason:</strong> {item.fallbackReason}</p>}
          {item.explanation && <p><strong>Explanation:</strong> {item.explanation}</p>}
          {showRag && item.rag && (
            <p>
              <strong>RAG:</strong> {item.rag.verdict || '-'} | score: {item.rag.score ?? '-'}
              {item.rag.reason ? ` | ${item.rag.reason}` : ''}
            </p>
          )}
          {showScore && <p><strong>Quality Score:</strong> {item.qualityScore ?? '-'}</p>}
          {item.source && <p><strong>Source:</strong> {item.source}</p>}
          {item.modelOutput && (
            <details>
              <summary>Model Output (Raw)</summary>
              <pre>{item.modelOutput}</pre>
            </details>
          )}
        </article>
      ))}
    </div>
  )
}

function RawOutputList({ rawOutputs = [] }) {
  if (!rawOutputs?.length) {
    return <p className="muted">No raw model outputs.</p>
  }

  return (
    <div className="question-list">
      <h3>Actual Finetune Model Outputs</h3>
      {rawOutputs.map((item) => (
        <article className="question-card" key={`raw-${item.questionIndex}-${item.chunkIndex}`}>
          <p><strong>Question Index:</strong> {item.questionIndex}</p>
          <p><strong>Chunk Index:</strong> {item.chunkIndex}</p>
          <p><strong>Fallback Used:</strong> {item.usedFallback ? 'Yes' : 'No'}</p>
          {item.fallbackReason && <p><strong>Fallback Reason:</strong> {item.fallbackReason}</p>}
          <details>
            <summary>Show Model Prompt</summary>
            <pre>{item.prompt || ''}</pre>
          </details>
          <details>
            <summary>Show Model Output</summary>
            <pre>{item.raw || ''}</pre>
          </details>
        </article>
      ))}
    </div>
  )
}

export default App
