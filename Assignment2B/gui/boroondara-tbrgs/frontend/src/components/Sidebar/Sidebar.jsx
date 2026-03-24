import { useState } from 'react';
import { SCATS_SITES, ML_MODELS, DEFAULTS } from '../../utils/constants';
import './Sidebar.css';

export default function Sidebar({ onSearch, loading }) {
  const [origin,      setOrigin]      = useState('2000');
  const [destination, setDestination] = useState('3002');
  const [model,       setModel]       = useState(DEFAULTS.model);
  const [departTime,  setDepartTime]  = useState('08:30');
  const [topK,        setTopK]        = useState(DEFAULTS.topK);

  const canSearch = origin !== destination;

  const handleSubmit = () => {
    if (!canSearch || loading) return;
    onSearch({ origin, destination, model, departTime, topK });
  };

  const swap = () => {
    setOrigin(destination);
    setDestination(origin);
  };

  return (
    <aside className="sidebar">

      {/* Journey */}
      <div className="sb-section">
        <p className="sb-label">Journey</p>
        <div className="journey-wrap">
          <div className="journey-field">
            <span className="jdot jdot--origin" />
            <div className="jfield-inner">
              <label className="jfield-tag">Origin</label>
              <select value={origin} onChange={e => setOrigin(e.target.value)}>
                {SCATS_SITES.map(s => (
                  <option key={s.id} value={s.id}>{s.id} — {s.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="journey-middle">
            <div className="journey-line" />
            <button className="swap-btn" onClick={swap} title="Swap origin and destination">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M3 1v10M3 11L1 9M3 11l2-2M9 11V1M9 1L7 3M9 1l2 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <div className="journey-line" />
          </div>

          <div className="journey-field">
            <span className="jdot jdot--dest" />
            <div className="jfield-inner">
              <label className="jfield-tag">Destination</label>
              <select value={destination} onChange={e => setDestination(e.target.value)}>
                {SCATS_SITES.map(s => (
                  <option key={s.id} value={s.id}>{s.id} — {s.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
        {!canSearch && <p className="field-error">Origin and destination must differ</p>}
      </div>

      {/* Parameters */}
      <div className="sb-section">
        <p className="sb-label">Parameters</p>
        <div className="param-grid">
          <div className="param-tile">
            <span className="param-val">60 km/h</span>
            <span className="param-key">Speed limit</span>
          </div>
          <div className="param-tile">
            <span className="param-val">30 s</span>
            <span className="param-key">Intersection delay</span>
          </div>
          <div className="param-tile param-tile--input">
            <input
              type="time"
              value={departTime}
              onChange={e => setDepartTime(e.target.value)}
              className="param-input"
            />
            <span className="param-key">Depart time</span>
          </div>
          <div className="param-tile param-tile--input">
            <select
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              className="param-input"
            >
              {[1,2,3,4,5].map(n => <option key={n} value={n}>{n} route{n > 1 ? 's' : ''}</option>)}
            </select>
            <span className="param-key">Top-k routes</span>
          </div>
        </div>
      </div>

      {/* Model */}
      <div className="sb-section sb-section--grow">
        <p className="sb-label">Prediction model</p>
        <div className="model-list">
          {ML_MODELS.map(m => (
            <button
              key={m.id}
              className={`model-row${model === m.id ? ' model-row--active' : ''}`}
              onClick={() => setModel(m.id)}
              style={{ '--mc': m.color }}
            >
              <span className="model-radio">
                {model === m.id && <span className="model-radio-inner" />}
              </span>
              <span className="model-text">
                <span className="model-name">{m.label}</span>
                <span className="model-sub">{m.fullName}</span>
              </span>
              <span className="model-rmse">RMSE {m.rmse}</span>
            </button>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="sb-footer">
        <button
          className={`find-btn${loading ? ' find-btn--loading' : ''}${!canSearch ? ' find-btn--disabled' : ''}`}
          onClick={handleSubmit}
          disabled={loading || !canSearch}
        >
          {loading
            ? <><span className="btn-spinner" />Calculating…</>
            : 'Find routes'}
        </button>
      </div>

    </aside>
  );
}
