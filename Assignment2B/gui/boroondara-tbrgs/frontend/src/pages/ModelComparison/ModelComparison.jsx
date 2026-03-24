import { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts';
import StatCard from '../../components/StatCard/StatCard';
import { MOCK_COMPARISON } from '../../utils/mockData';
import { ML_MODELS } from '../../utils/constants';
import './ModelComparison.css';

const TABS = ['Overview', 'Training Loss', 'Validation Loss', 'Radar'];

export default function ModelComparison() {
  const [tab,  setTab]  = useState('Overview');
  const [data, setData] = useState(null);

  useEffect(() => {
    // Simulate fetch — swap for real API call when backend is ready
    setTimeout(() => setData(MOCK_COMPARISON), 400);
  }, []);

  if (!data) return (
    <div className="mc-loading">
      <span className="mc-spinner" />
      <span>Loading model metrics…</span>
    </div>
  );

  const { metrics, trainingLoss, valLoss } = data;

  // Build bar chart data
  const barData = [
    { name: 'RMSE', LSTM: metrics.rmse[0], GRU: metrics.rmse[1], Transformer: metrics.rmse[2] },
    { name: 'MAE',  LSTM: metrics.mae[0],  GRU: metrics.mae[1],  Transformer: metrics.mae[2]  },
    { name: 'MAPE', LSTM: metrics.mape[0], GRU: metrics.mape[1], Transformer: metrics.mape[2] },
  ];

  const radarData = [
    { metric: 'RMSE',     LSTM: 100 - metrics.rmse[0] * 4, GRU: 100 - metrics.rmse[1] * 4, Transformer: 100 - metrics.rmse[2] * 4 },
    { metric: 'MAE',      LSTM: 100 - metrics.mae[0]  * 5, GRU: 100 - metrics.mae[1]  * 5, Transformer: 100 - metrics.mae[2]  * 5 },
    { metric: 'MAPE',     LSTM: 100 - metrics.mape[0] * 5, GRU: 100 - metrics.mape[1] * 5, Transformer: 100 - metrics.mape[2] * 5 },
    { metric: 'R²',       LSTM: metrics.r2[0] * 100,       GRU: metrics.r2[1] * 100,       Transformer: metrics.r2[2] * 100       },
    { metric: 'Speed',    LSTM: 78,                         GRU: 85,                         Transformer: 62                        },
    { metric: 'Stability',LSTM: 82,                         GRU: 88,                         Transformer: 76                        },
  ];

  const tooltipStyle = {
    contentStyle: { background: 'var(--surface)', border: '0.5px solid var(--border-md)', borderRadius: 8, fontSize: 12 },
    labelStyle:   { color: 'var(--text-2)', fontWeight: 500 },
  };

  const bestModel = ML_MODELS.reduce((best, m) =>
    m.rmse < best.rmse ? m : best
  );

  return (
    <div className="mc-page">

      {/* Header */}
      <div className="mc-header">
        <div>
          <h1 className="mc-title">Model Comparison</h1>
          <p className="mc-sub">Comparing LSTM, GRU and Transformer on Boroondara traffic data</p>
        </div>
        <div className="mc-best-tag">
          <span className="mc-best-dot" style={{ background: bestModel.color }} />
          <span>{bestModel.label} performs best</span>
        </div>
      </div>

      {/* Stat row */}
      <div className="mc-stats">
        {ML_MODELS.map(m => (
          <div key={m.id} className="mc-model-stats">
            <div className="mc-model-header" style={{ '--mc': m.color }}>
              <span className="mc-model-dot" />
              <span className="mc-model-name">{m.label}</span>
              <span className="mc-model-full">{m.fullName}</span>
            </div>
            <div className="mc-metric-grid">
              <StatCard small label="RMSE" value={m.rmse} accent={m.color} />
              <StatCard small label="MAE"  value={m.mae}  accent={m.color} />
              <StatCard small label="MAPE" value={`${m.mape}%`} accent={m.color} />
              <StatCard small label="R²"   value={m.r2}   accent={m.color} />
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="mc-tabs">
        {TABS.map(t => (
          <button
            key={t}
            className={`mc-tab${tab === t ? ' mc-tab--active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Charts */}
      <div className="mc-chart-area">

        {tab === 'Overview' && (
          <div className="mc-chart-wrap">
            <p className="chart-title">Error metrics comparison (lower is better)</p>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={barData} barGap={4} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-2)' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-3)' }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
                {ML_MODELS.map(m => (
                  <Bar key={m.id} dataKey={m.label} fill={m.color} radius={[4,4,0,0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {tab === 'Training Loss' && (
          <div className="mc-chart-wrap">
            <p className="chart-title">Training loss over 60 epochs</p>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={trainingLoss}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-3)' }} axisLine={false} tickLine={false} label={{ value: 'Epoch', position: 'insideBottomRight', offset: -8, fontSize: 11, fill: 'var(--text-3)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-3)' }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
                {ML_MODELS.map(m => (
                  <Line key={m.id} type="monotone" dataKey={m.id} name={m.label} stroke={m.color} strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {tab === 'Validation Loss' && (
          <div className="mc-chart-wrap">
            <p className="chart-title">Validation loss over 60 epochs</p>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={valLoss}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-3)' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-3)' }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
                {ML_MODELS.map(m => (
                  <Line key={m.id} type="monotone" dataKey={m.id} name={m.label} stroke={m.color} strokeWidth={2} dot={false} strokeDasharray={m.id === 'transformer' ? '5 3' : undefined} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {tab === 'Radar' && (
          <div className="mc-chart-wrap">
            <p className="chart-title">Multi-dimensional model evaluation (higher is better)</p>
            <ResponsiveContainer width="100%" height={360}>
              <RadarChart data={radarData} margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                <PolarGrid stroke="var(--border-md)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: 'var(--text-2)' }} />
                <Tooltip {...tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {ML_MODELS.map(m => (
                  <Radar key={m.id} name={m.label} dataKey={m.label} stroke={m.color} fill={m.color} fillOpacity={0.12} strokeWidth={2} />
                ))}
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

      </div>
    </div>
  );
}
