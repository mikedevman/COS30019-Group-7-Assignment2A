import { useState } from 'react';
import './Settings.css';

const DEFAULT_CONFIG = {
  speedLimit:        60,
  intersectionDelay: 30,
  topK:              5,
  defaultModel:      'lstm',
  epochs:            60,
  batchSize:         32,
  learningRate:      0.001,
  sequenceLen:       12,
  hiddenUnits:       64,
};

export default function Settings() {
  const [cfg, setCfg]     = useState(DEFAULT_CONFIG);
  const [saved, setSaved] = useState(false);

  const set = (key, val) => {
    setCfg(prev => ({ ...prev, [key]: val }));
    setSaved(false);
  };

  const handleSave = () => {
    localStorage.setItem('tbrgs_config', JSON.stringify(cfg));
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  const handleReset = () => {
    setCfg(DEFAULT_CONFIG);
    localStorage.removeItem('tbrgs_config');
    setSaved(false);
  };

  return (
    <div className="settings-page">
      <div className="settings-inner">

        <div className="settings-head">
          <h1 className="settings-title">Settings</h1>
          <p className="settings-sub">Configure TBRGS defaults</p>
        </div>

        {/* Route settings */}
        <section className="settings-section">
          <h2 className="settings-section-title">Route parameters</h2>
          <div className="settings-grid">
            <div className="setting-row">
              <div className="setting-label">
                <span>Speed limit</span>
                <span className="setting-hint">Applied to all road segments</span>
              </div>
              <div className="setting-control">
                <input type="number" value={cfg.speedLimit} min={20} max={110}
                  onChange={e => set('speedLimit', Number(e.target.value))} />
                <span className="setting-unit">km/h</span>
              </div>
            </div>
            <div className="setting-row">
              <div className="setting-label">
                <span>Intersection delay</span>
                <span className="setting-hint">Average delay per controlled intersection</span>
              </div>
              <div className="setting-control">
                <input type="number" value={cfg.intersectionDelay} min={0} max={120}
                  onChange={e => set('intersectionDelay', Number(e.target.value))} />
                <span className="setting-unit">seconds</span>
              </div>
            </div>
            <div className="setting-row">
              <div className="setting-label">
                <span>Top-K routes</span>
                <span className="setting-hint">Maximum routes returned per query</span>
              </div>
              <div className="setting-control">
                <input type="number" value={cfg.topK} min={1} max={10}
                  onChange={e => set('topK', Number(e.target.value))} />
              </div>
            </div>
            <div className="setting-row">
              <div className="setting-label">
                <span>Default model</span>
                <span className="setting-hint">ML model used by default on load</span>
              </div>
              <div className="setting-control">
                <select value={cfg.defaultModel} onChange={e => set('defaultModel', e.target.value)}>
                  <option value="lstm">LSTM</option>
                  <option value="gru">GRU</option>
                  <option value="transformer">Transformer</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* ML hyperparameters */}
        <section className="settings-section">
          <h2 className="settings-section-title">ML hyperparameters</h2>
          <p className="settings-section-sub">These values are sent to the Python ML service when triggering training</p>
          <div className="settings-grid">
            {[
              { key: 'epochs',       label: 'Epochs',          hint: 'Training iterations' },
              { key: 'batchSize',    label: 'Batch size',      hint: 'Samples per gradient step' },
              { key: 'learningRate', label: 'Learning rate',   hint: 'Optimiser step size' },
              { key: 'sequenceLen', label: 'Sequence length', hint: 'Input time steps (× 15 min)' },
              { key: 'hiddenUnits', label: 'Hidden units',    hint: 'LSTM/GRU hidden layer size' },
            ].map(({ key, label, hint }) => (
              <div className="setting-row" key={key}>
                <div className="setting-label">
                  <span>{label}</span>
                  <span className="setting-hint">{hint}</span>
                </div>
                <div className="setting-control">
                  <input
                    type="number"
                    value={cfg[key]}
                    step={key === 'learningRate' ? 0.0001 : 1}
                    min={0}
                    onChange={e => set(key, key === 'learningRate' ? parseFloat(e.target.value) : Number(e.target.value))}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Actions */}
        <div className="settings-actions">
          <button className="settings-btn settings-btn--reset" onClick={handleReset}>
            Reset to defaults
          </button>
          <button className={`settings-btn settings-btn--save${saved ? ' settings-btn--saved' : ''}`} onClick={handleSave}>
            {saved ? '✓ Saved' : 'Save settings'}
          </button>
        </div>

      </div>
    </div>
  );
}
