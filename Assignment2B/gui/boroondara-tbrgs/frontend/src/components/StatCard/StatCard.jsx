import './StatCard.css';

export default function StatCard({ label, value, unit, accent, small }) {
  return (
    // Reusable stat tile with optional accent bar and compact variant
    <div className={`stat-card${small ? ' stat-card--small' : ''}`} style={accent ? { '--sa': accent } : {}}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </span>
      {accent && <div className="stat-bar" />}
    </div>
  );
}
