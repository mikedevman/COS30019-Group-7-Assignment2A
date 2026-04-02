import { useState } from 'react';
import { SCATS_SITES, ML_MODELS, DEFAULTS } from '../../utils/constants';
import './Sidebar.css';

export default function Sidebar({ onSearch, loading }) {
  const [origin,      setOrigin]      = useState(DEFAULTS.origin);
  const [destination, setDestination] = useState(DEFAULTS.destination);
  const [model,       setModel]       = useState(DEFAULTS.model);
  const [departTime,  setDepartTime]  = useState(DEFAULTS.departTime);
  const [topK,        setTopK]        = useState(DEFAULTS.topK);
  const [speedLimit,  setSpeedLimit]  = useState(DEFAULTS.speedLimit);
  const [intersectionDelay, setIntersectionDelay] = useState(DEFAULTS.intersectionDelay);
  const [isbidirectional, setIsbidirectional] = useState(false);

  const canSearch = origin !== destination;

  const handleSubmit = () => {
    if (!canSearch || loading) return;
    onSearch({
      origin,
      destination,
      model,
      departTime,
      topK,
      speedLimit,
      intersectionDelay,
      bidirectional: (model === 'lstm' || model === 'gru') ? isbidirectional : false,
    });
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
          <div className="param-tile param-tile--input">
            <input
              type="number"
              value={speedLimit}
              onChange={e => setSpeedLimit(Number(e.target.value))}
              className="param-input"
              min="10" max="130" step="5"
            />
            <span className="param-key">Speed (km/h)</span>
          </div>
          <div className="param-tile param-tile--input">
            <input
              type="number"
              value={intersectionDelay}
              onChange={e => setIntersectionDelay(Number(e.target.value))}
              className="param-input"
              min="0" max="120" step="5"
            />
            <span className="param-key">Delay (s)</span>
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
              onClick={() => {
                setModel(m.id);
                if (m.id !== 'lstm' && m.id !== 'gru') setIsbidirectional(false);
              }}
              style={{ '--mc': m.color }}
            >
              <span className="model-radio">
                {model === m.id && <span className="model-radio-inner" />}
              </span>
              <span className="model-text">
                <span className="model-name">{m.label}</span>
                <span className="model-sub">{m.fullName}</span>
              </span>

              {(m.id === 'lstm' || m.id === 'gru') && model === m.id && (
                <div className={`bidirectional-inline-toggle${isbidirectional ? ' bidirectional-inline-toggle--active' : ''}`}>
                  <label className="bidirectional-upgrade-label" title={`Enabling this will switch to Bidirectional ${m.label}`}>
                    <span className="bidirectional-upgrade-text">
                      <span className="bidirectional-upgrade-title">Bidirectional</span>
                    </span>
                    <span className="bidirectional-switch-row">
                      <input
                        type="checkbox"
                        checked={isbidirectional}
                        onChange={e => {
                          e.stopPropagation();
                          setIsbidirectional(prev => !prev);
                        }}
                        disabled={loading}
                      />
                      <span className="bidirectional-switch-ui" aria-hidden="true" />
                    </span>
                  </label>
                  <span className={`bidirectional-status${isbidirectional ? ' bidirectional-status--on' : ' bidirectional-status--off'}`}>
                    {isbidirectional ? 'On' : 'Off'}
                  </span>
                </div>
              )}
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
