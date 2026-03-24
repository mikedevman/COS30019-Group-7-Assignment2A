import { ML_MODELS } from './constants';

const rng = (min, max) => Math.round(min + Math.random() * (max - min));

export const MOCK_ROUTES = [
  { id: 1, duration: 14, distance: 6.2, via: 'Toorak Rd -> Burke Rd', intersections: 4, path: ['2000', '3001', '3003', '3002'], best: true },
  { id: 2, duration: 17, distance: 5.9, via: 'Warrigal Rd -> High St', intersections: 3, path: ['2000', '2001', '3003', '3002'], best: false },
  { id: 3, duration: 19, distance: 7.1, via: 'High St -> Burke Rd', intersections: 5, path: ['2000', '2001', '2002', '3004', '3002'], best: false },
  { id: 4, duration: 21, distance: 7.8, via: 'Whitehorse Rd -> Balwyn Rd', intersections: 6, path: ['2000', '3001', '4002', '4001', '3002'], best: false },
  { id: 5, duration: 24, distance: 8.4, via: 'Canterbury Rd -> Denmark St', intersections: 7, path: ['2000', '2002', '5001', '4003', '3002'], best: false },
];

export const MOCK_COMPARISON = {
  models: ML_MODELS.map((m) => m.label),
  metrics: {
    rmse: ML_MODELS.map((m) => m.rmse),
    mae: ML_MODELS.map((m) => m.mae),
    mape: ML_MODELS.map((m) => m.mape),
    r2: ML_MODELS.map((m) => m.r2),
  },
  trainingLoss: Array.from({ length: 60 }, (_, i) => ({
    epoch: i + 1,
    lstm: parseFloat(Math.max(0.04, 0.82 * Math.exp(-i * 0.08) + (Math.random() - 0.5) * 0.015).toFixed(4)),
    gru: parseFloat(Math.max(0.05, 0.88 * Math.exp(-i * 0.075) + (Math.random() - 0.5) * 0.015).toFixed(4)),
    transformer: parseFloat(Math.max(0.03, 0.76 * Math.exp(-i * 0.09) + (Math.random() - 0.5) * 0.012).toFixed(4)),
  })),
  valLoss: Array.from({ length: 60 }, (_, i) => ({
    epoch: i + 1,
    lstm: parseFloat(Math.max(0.06, 0.9 * Math.exp(-i * 0.07) + (Math.random() - 0.5) * 0.02).toFixed(4)),
    gru: parseFloat(Math.max(0.07, 0.95 * Math.exp(-i * 0.065) + (Math.random() - 0.5) * 0.02).toFixed(4)),
    transformer: parseFloat(Math.max(0.05, 0.84 * Math.exp(-i * 0.08) + (Math.random() - 0.5) * 0.018).toFixed(4)),
  })),
};

export const MOCK_FORECAST = Array.from({ length: 96 }, (_, i) => {
  const hour = (i * 15) / 60;
  const peak = (h) =>
    120 * Math.exp(-0.5 * Math.pow((h - 8.5) / 1.2, 2)) +
    100 * Math.exp(-0.5 * Math.pow((h - 17.5) / 1.5, 2));
  const base = 40 + peak(hour);
  const hh = String(Math.floor(hour)).padStart(2, '0');
  const mm = String((i * 15) % 60).padStart(2, '0');
  return {
    time: `${hh}:${mm}`,
    actual: Math.max(0, rng(base - 10, base + 10)),
    predicted: Math.max(0, rng(base - 6, base + 8)),
  };
});
