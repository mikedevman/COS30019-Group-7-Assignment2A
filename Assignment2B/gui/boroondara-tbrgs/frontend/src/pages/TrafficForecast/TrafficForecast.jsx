import { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { SCATS_SITES, ML_MODELS } from '../../utils/constants';
import { MOCK_FORECAST } from '../../utils/mockData';
import StatCard from '../../components/StatCard/StatCard';
import './TrafficForecast.css';

function getNowIndex() {
  const now = new Date();
  return Math.floor((now.getHours() * 60 + now.getMinutes()) / 15);
}

export default function TrafficForecast() {
  const [siteId,  setSiteId]  = useState('2000');
  const [modelId, setModelId] = useState('lstm');
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(false);

  const loadData = (site, model) => {
    setLoading(true);
    setTimeout(() => {
      setData(MOCK_FORECAST);
      setLoading(false);
    }, 500);
  };

  useEffect(() => { loadData(siteId, modelId); }, [siteId, modelId]);

  const nowIdx   = getNowIndex();
  const nowLabel = data ? data[Math.min(nowIdx, data.length - 1)]?.time : null;

  const peak    = data ? Math.max(...data.map(d => d.actual))    : 0;
  const current = data ? (data[Math.min(nowIdx, data.length-1)]?.actual ?? 0) : 0;
  const avg     = data ? Math.round(data.reduce((s, d) => s + d.actual, 0) / data.length) : 0;
  const accuracy = data
    ? (100 - (data.reduce((s, d) => s + Math.abs(d.actual - d.predicted) / (d.actual || 1), 0) / data.length * 100)).toFixed(1)
    : 0;

  const model = ML_MODELS.find(m => m.id === modelId);
  const site  = SCATS_SITES.find(s => s.id === siteId);

  // Show only every 4th tick (hourly) on x-axis
  const tickFormatter = (val, idx) => idx % 4 === 0 ? val : '';

  const tooltipStyle = {
    contentStyle: { background: 'var(--surface)', border: '0.5px solid var(--border-md)', borderRadius: 8, fontSize: 12 },
    labelStyle: { color: 'var(--text-2)', fontWeight: 500, marginBottom: 4 },
  };

  return (
    <div className="tf-page">
      {/* Header */}
      <div className="tf-header">
        <div>
          <h1 className="tf-title">Traffic Forecast</h1>
          <p className="tf-sub">Predicted vs actual traffic volume at SCATS sites</p>
        </div>
        <div className="tf-controls">
          <div className="tf-ctrl-group">
            <label>SCATS site</label>
            <select value={siteId} onChange={e => setSiteId(e.target.value)}>
              {SCATS_SITES.map(s => (
                <option key={s.id} value={s.id}>{s.id} — {s.name}</option>
              ))}
            </select>
          </div>
          <div className="tf-ctrl-group">
            <label>Model</label>
            <select value={modelId} onChange={e => setModelId(e.target.value)}>
              {ML_MODELS.map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="tf-stats">
        <StatCard label="Current volume" value={current} unit="veh/15min" accent={model?.color} />
        <StatCard label="Daily peak"     value={peak}    unit="veh/15min" accent={model?.color} />
        <StatCard label="Daily average"  value={avg}     unit="veh/15min" accent={model?.color} />
        <StatCard label="Model accuracy" value={`${accuracy}%`} accent={model?.color} />
      </div>

      {/* Chart */}
      <div className="tf-chart-card">
        <div className="tf-chart-header">
          <div>
            <p className="tf-chart-title">24-hour traffic volume — {site?.name}</p>
            <p className="tf-chart-sub">Interval: 15 min &nbsp;·&nbsp; Model: {model?.label} &nbsp;·&nbsp; Date: Oct 2006</p>
          </div>
          <div className="tf-legend-custom">
            <span className="tfl-item">
              <span className="tfl-dot" style={{ background: 'var(--green)' }} />
              Actual
            </span>
            <span className="tfl-item">
              <span className="tfl-dash" style={{ background: model?.color }} />
              Predicted
            </span>
          </div>
        </div>

        {loading ? (
          <div className="tf-chart-loading">
            <span className="tf-spinner" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gradActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#1D9E75" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#1D9E75" stopOpacity={0}    />
                </linearGradient>
                <linearGradient id="gradPred" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={model?.color} stopOpacity={0.12} />
                  <stop offset="95%" stopColor={model?.color} stopOpacity={0}    />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 11, fill: 'var(--text-3)' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={tickFormatter}
              />
              <YAxis
                tick={{ fontSize: 11, fill: 'var(--text-3)' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'vehicles', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: 'var(--text-3)' }}
              />
              <Tooltip {...tooltipStyle} />
              {nowLabel && (
                <ReferenceLine
                  x={nowLabel}
                  stroke="var(--amber)"
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{ value: 'Now', position: 'top', fontSize: 11, fill: 'var(--amber)' }}
                />
              )}
              <Area
                type="monotone"
                dataKey="actual"
                name="Actual"
                stroke="#1D9E75"
                strokeWidth={2}
                fill="url(#gradActual)"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="predicted"
                name="Predicted"
                stroke={model?.color}
                strokeWidth={2}
                strokeDasharray="5 3"
                fill="url(#gradPred)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Interval table - last 8 readings */}
      {data && (
        <div className="tf-table-card">
          <p className="tf-table-title">Recent 15-min intervals</p>
          <table className="tf-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Actual (veh)</th>
                <th>Predicted (veh)</th>
                <th>Error</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.slice(Math.max(0, nowIdx - 7), nowIdx + 1).reverse().map((row, i) => {
                const err = Math.abs(row.actual - row.predicted);
                const pct = row.actual ? ((err / row.actual) * 100).toFixed(1) : '—';
                const status = err < 8 ? 'good' : err < 16 ? 'ok' : 'poor';
                return (
                  <tr key={i}>
                    <td className="tf-td-mono">{row.time}</td>
                    <td className="tf-td-mono">{row.actual}</td>
                    <td className="tf-td-mono">{row.predicted}</td>
                    <td className="tf-td-mono">{pct}%</td>
                    <td><span className={`tf-status tf-status--${status}`}>{status}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
