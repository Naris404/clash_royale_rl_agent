import type { Lang, Strings } from "../i18n";

interface Props {
  lang: Lang;
  strings: Strings;
  onLang: (lang: Lang) => void;
  onPlay: (mode: string) => void;
  onDashboard: () => void;
  busy: boolean;
}

const MODES = [
  { id: "coach", icon: "🧠", titleKey: "mode_coach", descKey: "mode_coach_desc" },
  { id: "vs_rl", icon: "🤖", titleKey: "mode_vs_rl", descKey: "mode_vs_rl_desc" },
  { id: "vs_bot", icon: "⚔️", titleKey: "mode_vs_bot", descKey: "mode_vs_bot_desc" },
] as const;

export function HomeScreen({ lang, strings, onLang, onPlay, onDashboard, busy }: Props) {
  const experiments = strings.research_items.split("|");

  return (
    <div className="home">
      <header className="home-header">
        <span className="hero-eyebrow">{strings.portfolio_eyebrow}</span>
        <h1>{strings.title}</h1>
        <p className="home-subtitle">{strings.subtitle}</p>
        <p className="hero-intro">{strings.portfolio_intro}</p>
        <button className="btn dashboard-hero-link" onClick={onDashboard}>
          ◫ {strings.dashboard}
        </button>
        <div className="lang-switch">
          <button aria-label="Polski" className={`btn btn-small ${lang === "pl" ? "btn-active" : ""}`} onClick={() => onLang("pl")}>
            PL
          </button>
          <button aria-label="English" className={`btn btn-small ${lang === "en" ? "btn-active" : ""}`} onClick={() => onLang("en")}>
            EN
          </button>
        </div>
      </header>

      <section className="portfolio-section" aria-labelledby="abstract-heading">
        <div className="section-heading">
          <span>{strings.about_title}</span>
          <h2 id="abstract-heading">{strings.thesis_abstract_title}</h2>
        </div>
        <p className="abstract-copy">{strings.thesis_abstract}</p>
      </section>

      <section className="portfolio-section" aria-labelledby="architecture-heading">
        <div className="section-heading">
          <h2 id="architecture-heading">{strings.architecture_title}</h2>
        </div>
        <div className="architecture-grid">
          {([
            ["01", strings.architecture_rl, strings.architecture_rl_desc],
            ["02", strings.architecture_coach, strings.architecture_coach_desc],
            ["03", strings.architecture_live, strings.architecture_live_desc],
          ] as const).map(([number, title, description]) => (
            <article className="architecture-card" key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="research-strip" aria-label={strings.research_title}>
        <div>
          <h2>{strings.research_title}</h2>
          <ul>{experiments.map((item) => <li key={item}>{item}</li>)}</ul>
        </div>
        <div className="stack-block">
          <h2>{strings.stack_title}</h2>
          <p>{strings.stack}</p>
        </div>
      </section>

      <section className="demo-section" aria-labelledby="demo-heading">
        <span className="hero-eyebrow">{strings.try_demo}</span>
        <h2 id="demo-heading">{strings.choose_mode}</h2>
      <div className="mode-grid">
        {MODES.map((mode) => (
          <button key={mode.id} className="mode-card" onClick={() => onPlay(mode.id)} disabled={busy}>
            <span className="mode-icon">{mode.icon}</span>
            <span className="mode-title">{strings[mode.titleKey]}</span>
            <span className="mode-desc">{strings[mode.descKey]}</span>
            <span className="mode-cta">{strings.play} →</span>
          </button>
        ))}
      </div>
      </section>

      <footer className="home-footer">{strings.footer_thesis}</footer>
    </div>
  );
}
