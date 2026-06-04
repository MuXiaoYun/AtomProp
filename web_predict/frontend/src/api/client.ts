import axios from 'axios'

export interface PredictResult {
  smiles: string
  predicted_value: number
}

export interface HealthResponse {
  status: string
  device: string
  model_loaded: boolean
  model_name: string | null
}

const api = axios.create({
  baseURL: '/api',
  timeout: 600_000,
})

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health/')
  return data
}

export async function uploadModel(file: File): Promise<{ model_name: string }> {
  const form = new FormData()
  form.append('model', file)
  const { data } = await api.post('/model/upload/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function fetchModelStatus(): Promise<{
  loaded: boolean
  model_name: string | null
}> {
  const { data } = await api.get('/model/status/')
  return data
}

export async function predictSmiles(smiles: string[]): Promise<{
  results: PredictResult[]
  count: number
  input_count: number
}> {
  const { data } = await api.post('/predict/', { smiles })
  return data
}

export async function predictFile(file: File): Promise<{
  results: PredictResult[]
  count: number
  input_count: number
  source_file?: string
}> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/predict/file/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function exportResultsCsv(
  results: PredictResult[],
): Promise<Blob> {
  const { data } = await api.post('/export/', { results }, { responseType: 'blob' })
  return data
}

export function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const d = err.response?.data
    if (d && typeof d === 'object' && 'error' in d) return String((d as { error: string }).error)
    return err.message
  }
  if (err instanceof Error) return err.message
  return '未知错误'
}

// —— Training ——

export interface TrainDefaults {
  num_epochs: number
  learning_rate: number
  batch_size: number
  tolerance: number
}

export interface DatasetUploadResult {
  dataset_id: string
  filename: string
  smiles_column: string
  target_columns: string[]
  row_count: number
  preview: Record<string, string>[]
}

export interface TrainJobStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  epoch: number
  total_epochs: number
  train_losses: number[]
  val_losses: number[]
  message: string
  model_ready: boolean
  error: string | null
  task_type: string
  best_metric: number | null
}

export async function fetchTrainDefaults(): Promise<TrainDefaults> {
  const { data } = await api.get<TrainDefaults>('/train/defaults/')
  return data
}

export async function uploadTrainDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<DatasetUploadResult>('/train/dataset/upload/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function uploadTrainInitModel(file: File): Promise<{ model_id: string; model_name: string }> {
  const form = new FormData()
  form.append('model', file)
  const { data } = await api.post('/train/init-model/upload/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function startTraining(payload: {
  dataset_id: string
  target_column: string
  task_type: string
  init_mode: string
  init_model_id?: string
  num_epochs: number
  learning_rate: number
}): Promise<{ job_id: string }> {
  const { data } = await api.post('/train/start/', payload)
  return data
}

export async function pollTrainJob(jobId: string): Promise<TrainJobStatus> {
  const { data } = await api.get<TrainJobStatus>(`/train/jobs/${jobId}/`)
  return data
}

export async function downloadTrainedModel(jobId: string): Promise<Blob> {
  const { data } = await api.get(`/train/jobs/${jobId}/model/`, { responseType: 'blob' })
  return data
}
