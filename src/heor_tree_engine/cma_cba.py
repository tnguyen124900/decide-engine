"""Cost-minimization (CMA) and cost-benefit (CBA) analysis.

Pure functions shared by every DECIDE model type (decision tree, Markov,
PSM, microsimulation, DES).  They take per-strategy expected totals that the
model engines already produce (costs and effects, already discounted with the
model's own discount rates) and return plain rows, so the UI, the reports and
the tests all use the same numbers.

CMA  -- only valid when the strategies' outcomes are equivalent. The function
        ranks strategies by expected cost, flags the cheapest and reports how
        far the effects differ so the caller can warn.
CBA  -- every outcome is expressed in money. Benefit = (money value of one
        effect unit x effect, sign-adjusted by the effect objective) + any
        extra monetary benefit entered per strategy (e.g. productivity gain,
        avoided costs outside the model).  Net benefit (NPV) = benefit - cost;
        benefit-cost ratio (BCR) and ROI are computed against the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


# --------------------------------------------------------------------------
# Cost-minimization
# --------------------------------------------------------------------------
@dataclass
class CmaRow:
    name: str
    cost: float
    effect: float
    extra_cost_vs_cheapest: float
    is_cheapest: bool


@dataclass
class CmaResult:
    rows: list[CmaRow]
    cheapest: str
    effect_spread: float  # (max - min) / max(|mean|, tiny), as a fraction
    effects_equivalent: bool
    tolerance: float
    ties: list[str] = field(default_factory=list)


def cma_analysis(
    names: Sequence[str],
    costs: Sequence[float],
    effects: Sequence[float],
    tolerance: float = 0.01,
) -> CmaResult:
    """Rank strategies by cost. ``tolerance`` is the relative spread in
    effect (fraction of the mean absolute effect) still treated as
    "equivalent"."""

    if not names or not (len(names) == len(costs) == len(effects)):
        raise ValueError("names, costs and effects must be non-empty and the same length")
    costs = [float(c) for c in costs]
    effects = [float(e) for e in effects]
    low = min(costs)
    eps = max(1e-9, 1e-9 * abs(low))
    rows = [
        CmaRow(
            name=n, cost=c, effect=e,
            extra_cost_vs_cheapest=c - low,
            is_cheapest=(c - low) <= eps,
        )
        for n, c, e in zip(names, costs, effects)
    ]
    cheapest_rows = [r for r in rows if r.is_cheapest]
    mean_abs = sum(abs(e) for e in effects) / len(effects)
    spread = (max(effects) - min(effects)) / mean_abs if mean_abs > 1e-12 else 0.0
    return CmaResult(
        rows=rows,
        cheapest=cheapest_rows[0].name,
        effect_spread=spread,
        effects_equivalent=spread <= tolerance + 1e-12,
        tolerance=tolerance,
        ties=[r.name for r in cheapest_rows[1:]],
    )


# --------------------------------------------------------------------------
# Cost-benefit
# --------------------------------------------------------------------------
@dataclass
class CbaRow:
    name: str
    cost: float
    effect: float
    benefit: float  # total monetary benefit
    net_benefit: float  # benefit - cost  (NPV; inputs are already discounted)
    delta_cost: float | None  # vs baseline
    delta_benefit: float | None
    delta_net_benefit: float | None
    bcr: float | None  # delta_benefit / delta_cost
    roi: float | None  # delta_net_benefit / delta_cost
    bcr_note: str = ""  # "" | "baseline" | "dominant" | "dominated" | "no_cost_change"


@dataclass
class CbaResult:
    rows: list[CbaRow]
    baseline: str
    best: str  # highest net benefit
    value_per_effect: float


def cba_analysis(
    names: Sequence[str],
    costs: Sequence[float],
    effects: Sequence[float],
    value_per_effect: float,
    maximize_effect: bool = True,
    extra_benefits: Mapping[str, float] | None = None,
    baseline: str | None = None,
) -> CbaResult:
    """Cost-benefit analysis. ``extra_benefits`` maps strategy name to an
    additional monetary benefit (already discounted); missing -> 0.
    ``baseline`` defaults to the cheapest strategy."""

    if not names or not (len(names) == len(costs) == len(effects)):
        raise ValueError("names, costs and effects must be non-empty and the same length")
    if len(set(names)) != len(names):
        raise ValueError("strategy names must be unique")
    extra_benefits = extra_benefits or {}
    sign = 1.0 if maximize_effect else -1.0
    costs = [float(c) for c in costs]
    effects = [float(e) for e in effects]
    benefits = [
        sign * float(value_per_effect) * e + float(extra_benefits.get(n, 0.0) or 0.0)
        for n, e in zip(names, effects)
    ]
    if baseline not in names:
        baseline = names[min(range(len(names)), key=lambda i: costs[i])]
    bi = list(names).index(baseline)

    rows: list[CbaRow] = []
    for i, n in enumerate(names):
        net = benefits[i] - costs[i]
        if i == bi:
            rows.append(CbaRow(n, costs[i], effects[i], benefits[i], net,
                               None, None, None, None, None, "baseline"))
            continue
        dc = costs[i] - costs[bi]
        db = benefits[i] - benefits[bi]
        dn = db - dc
        bcr = roi = None
        note = ""
        if abs(dc) < 1e-12:
            note = "no_cost_change"
        elif dc < 0:
            # cheaper than baseline: ratio is not meaningful
            note = "dominant" if db >= 0 else "cheaper_less_benefit"
        else:
            bcr = db / dc
            roi = dn / dc
            if db <= 0:
                note = "dominated"
        rows.append(CbaRow(n, costs[i], effects[i], benefits[i], net, dc, db, dn, bcr, roi, note))
    best = max(rows, key=lambda r: r.net_benefit).name
    return CbaResult(rows=rows, baseline=baseline, best=best, value_per_effect=float(value_per_effect))


# --------------------------------------------------------------------------
# Translations (merged into app.TRANSLATIONS)
# --------------------------------------------------------------------------
TEXT: dict[str, dict[str, str]] = {
    "en": {
        "analysis_type_cma": "Cost-Minimization Analysis (CMA)",
        "analysis_type_cba": "Cost-Benefit Analysis (CBA)",
        "analysis_type_cma_help": "Use only when strategies have equivalent outcomes: picks the cheapest strategy. DECIDE warns if the effects differ.",
        "analysis_type_cba_help": "Values everything in money: benefit = money value per effect unit x effect (+ extra benefits), then net benefit (NPV), benefit-cost ratio (BCR) and ROI vs the baseline. With the money value set to your willingness-to-pay, net benefit equals Net Monetary Benefit (NMB).",
        "cma_tolerance": "Effects treated as equivalent within (%)",
        "cma_cheapest": "Cheapest strategy: {name}",
        "cma_tie": "Tie on cost with: {names}",
        "cma_extra_cost": "Extra cost vs cheapest",
        "cma_cheapest_flag": "Cheapest",
        "cma_warning": "Effects differ by {spread:.1%} across strategies (more than the {tol:.1%} tolerance). Cost-minimization is only valid when outcomes are equivalent -- consider CEA instead.",
        "cma_ok": "Effects are equivalent within {tol:.1%} (spread {spread:.1%}); cost-minimization is appropriate.",
        "cba_value_per_effect": "Money value of one effect unit",
        "cba_extra_benefits": "Extra monetary benefits per strategy (already discounted; optional)",
        "cba_extra_benefit_for": "Extra benefit -- {name}",
        "cba_discount_note": "Costs and effects come from the model and are already discounted with the model's discount rates, so net benefit is a net present value (NPV).",
        "cba_benefit": "Total benefit",
        "cba_net_benefit": "Net benefit (NPV)",
        "cba_delta_cost": "Incremental cost",
        "cba_delta_benefit": "Incremental benefit",
        "cba_delta_net": "Incremental net benefit",
        "cba_bcr": "Benefit-cost ratio (BCR)",
        "cba_roi": "ROI",
        "cba_best": "Highest net benefit: {name}",
        "cba_note_baseline": "baseline",
        "cba_note_dominant": "cheaper and not worse",
        "cba_note_dominated": "costs more, no extra benefit",
        "cba_note_cheaper_less_benefit": "cheaper, less benefit",
        "cba_note_no_cost_change": "no cost change",
        "cba_baseline_info": "Comparison baseline: {name}",
    },
    "fr": {
        "analysis_type_cma": "Analyse de minimisation des coûts (CMA)",
        "analysis_type_cba": "Analyse coûts-bénéfices (CBA)",
        "analysis_type_cma_help": "À utiliser seulement si les résultats sont équivalents : retient la stratégie la moins chère. DECIDE avertit si les effets diffèrent.",
        "analysis_type_cba_help": "Tout est valorisé en argent : bénéfice = valeur monétaire par unité d'effet x effet (+ bénéfices additionnels), puis bénéfice net (VAN), ratio bénéfice-coût (BCR) et ROI par rapport à la référence.",
        "cma_tolerance": "Effets jugés équivalents à (%) près",
        "cma_cheapest": "Stratégie la moins chère : {name}",
        "cma_tie": "Égalité de coût avec : {names}",
        "cma_extra_cost": "Surcoût vs la moins chère",
        "cma_cheapest_flag": "Moins chère",
        "cma_warning": "Les effets diffèrent de {spread:.1%} entre les stratégies (au-delà de la tolérance de {tol:.1%}). La minimisation des coûts n'est valide que si les résultats sont équivalents -- envisagez une CEA.",
        "cma_ok": "Effets équivalents à {tol:.1%} près (écart {spread:.1%}) ; la minimisation des coûts est appropriée.",
        "cba_value_per_effect": "Valeur monétaire d'une unité d'effet",
        "cba_extra_benefits": "Bénéfices monétaires additionnels par stratégie (déjà actualisés ; facultatif)",
        "cba_extra_benefit_for": "Bénéfice additionnel -- {name}",
        "cba_discount_note": "Les coûts et effets proviennent du modèle et sont déjà actualisés avec ses taux d'actualisation ; le bénéfice net est donc une valeur actuelle nette (VAN).",
        "cba_benefit": "Bénéfice total",
        "cba_net_benefit": "Bénéfice net (VAN)",
        "cba_delta_cost": "Coût incrémental",
        "cba_delta_benefit": "Bénéfice incrémental",
        "cba_delta_net": "Bénéfice net incrémental",
        "cba_bcr": "Ratio bénéfice-coût (BCR)",
        "cba_roi": "ROI",
        "cba_best": "Bénéfice net le plus élevé : {name}",
        "cba_note_baseline": "référence",
        "cba_note_dominant": "moins chère et pas moins bonne",
        "cba_note_dominated": "plus chère, sans bénéfice additionnel",
        "cba_note_cheaper_less_benefit": "moins chère, moins de bénéfice",
        "cba_note_no_cost_change": "coût inchangé",
        "cba_baseline_info": "Référence de comparaison : {name}",
    },
    "es": {
        "analysis_type_cma": "Análisis de minimización de costos (CMA)",
        "analysis_type_cba": "Análisis costo-beneficio (CBA)",
        "analysis_type_cma_help": "Úselo solo si los resultados son equivalentes: elige la estrategia más barata. DECIDE advierte si los efectos difieren.",
        "analysis_type_cba_help": "Todo se valora en dinero: beneficio = valor monetario por unidad de efecto x efecto (+ beneficios adicionales); luego beneficio neto (VAN), razón beneficio-costo (BCR) y ROI frente a la referencia.",
        "cma_tolerance": "Efectos equivalentes dentro de (%)",
        "cma_cheapest": "Estrategia más barata: {name}",
        "cma_tie": "Empate en costo con: {names}",
        "cma_extra_cost": "Costo extra vs la más barata",
        "cma_cheapest_flag": "Más barata",
        "cma_warning": "Los efectos difieren {spread:.1%} entre estrategias (más que la tolerancia de {tol:.1%}). La minimización de costos solo es válida con resultados equivalentes -- considere un CEA.",
        "cma_ok": "Efectos equivalentes dentro de {tol:.1%} (diferencia {spread:.1%}); la minimización de costos es apropiada.",
        "cba_value_per_effect": "Valor monetario de una unidad de efecto",
        "cba_extra_benefits": "Beneficios monetarios adicionales por estrategia (ya descontados; opcional)",
        "cba_extra_benefit_for": "Beneficio adicional -- {name}",
        "cba_discount_note": "Costos y efectos provienen del modelo y ya están descontados con sus tasas de descuento, por lo que el beneficio neto es un valor actual neto (VAN).",
        "cba_benefit": "Beneficio total",
        "cba_net_benefit": "Beneficio neto (VAN)",
        "cba_delta_cost": "Costo incremental",
        "cba_delta_benefit": "Beneficio incremental",
        "cba_delta_net": "Beneficio neto incremental",
        "cba_bcr": "Razón beneficio-costo (BCR)",
        "cba_roi": "ROI",
        "cba_best": "Mayor beneficio neto: {name}",
        "cba_note_baseline": "referencia",
        "cba_note_dominant": "más barata y no peor",
        "cba_note_dominated": "cuesta más, sin beneficio extra",
        "cba_note_cheaper_less_benefit": "más barata, menor beneficio",
        "cba_note_no_cost_change": "sin cambio de costo",
        "cba_baseline_info": "Referencia de comparación: {name}",
    },
    "zh": {
        "analysis_type_cma": "成本最小化分析 (CMA)",
        "analysis_type_cba": "成本效益分析 (CBA)",
        "analysis_type_cma_help": "仅在各策略结局等效时使用：选出成本最低的策略。若效果不同，DECIDE 会给出警告。",
        "analysis_type_cba_help": "全部以货币计量：收益 = 每单位效果的货币价值 x 效果（+ 额外收益），再计算净收益 (NPV)、收益成本比 (BCR) 和相对基线的 ROI。",
        "cma_tolerance": "效果在以下范围内视为等效 (%)",
        "cma_cheapest": "成本最低的策略：{name}",
        "cma_tie": "成本并列：{names}",
        "cma_extra_cost": "相对最低成本的额外成本",
        "cma_cheapest_flag": "成本最低",
        "cma_warning": "各策略效果相差 {spread:.1%}（超过 {tol:.1%} 的容差）。成本最小化仅在结局等效时有效——请考虑改用 CEA。",
        "cma_ok": "效果在 {tol:.1%} 内等效（差异 {spread:.1%}）；适合使用成本最小化。",
        "cba_value_per_effect": "每单位效果的货币价值",
        "cba_extra_benefits": "各策略额外货币收益（已折现；可选）",
        "cba_extra_benefit_for": "额外收益 -- {name}",
        "cba_discount_note": "成本和效果来自模型，已按模型的折现率折现，因此净收益即净现值 (NPV)。",
        "cba_benefit": "总收益",
        "cba_net_benefit": "净收益 (NPV)",
        "cba_delta_cost": "增量成本",
        "cba_delta_benefit": "增量收益",
        "cba_delta_net": "增量净收益",
        "cba_bcr": "收益成本比 (BCR)",
        "cba_roi": "ROI",
        "cba_best": "净收益最高：{name}",
        "cba_note_baseline": "基线",
        "cba_note_dominant": "更便宜且不差",
        "cba_note_dominated": "成本更高且无额外收益",
        "cba_note_cheaper_less_benefit": "更便宜，收益更低",
        "cba_note_no_cost_change": "成本无变化",
        "cba_baseline_info": "比较基线：{name}",
    },
    "ko": {
        "analysis_type_cma": "비용최소화분석 (CMA)",
        "analysis_type_cba": "비용편익분석 (CBA)",
        "analysis_type_cma_help": "전략 간 결과가 동등할 때만 사용: 가장 저렴한 전략을 선택합니다. 효과가 다르면 DECIDE가 경고합니다.",
        "analysis_type_cba_help": "모두 금액으로 평가: 편익 = 효과 1단위의 금전 가치 x 효과 (+ 추가 편익). 이후 순편익(NPV), 편익-비용 비(BCR), 기준 대비 ROI를 계산합니다.",
        "cma_tolerance": "효과를 동등하게 보는 범위 (%)",
        "cma_cheapest": "가장 저렴한 전략: {name}",
        "cma_tie": "비용 동일: {names}",
        "cma_extra_cost": "최저가 대비 추가 비용",
        "cma_cheapest_flag": "최저가",
        "cma_warning": "전략 간 효과가 {spread:.1%} 차이납니다(허용 범위 {tol:.1%} 초과). 비용최소화는 결과가 동등할 때만 유효합니다 -- CEA를 고려하세요.",
        "cma_ok": "효과가 {tol:.1%} 이내로 동등합니다(차이 {spread:.1%}). 비용최소화가 적절합니다.",
        "cba_value_per_effect": "효과 1단위의 금전 가치",
        "cba_extra_benefits": "전략별 추가 금전 편익(이미 할인됨; 선택)",
        "cba_extra_benefit_for": "추가 편익 -- {name}",
        "cba_discount_note": "비용과 효과는 모델에서 가져오며 모델의 할인율로 이미 할인되어 있으므로 순편익은 순현재가치(NPV)입니다.",
        "cba_benefit": "총 편익",
        "cba_net_benefit": "순편익 (NPV)",
        "cba_delta_cost": "증분 비용",
        "cba_delta_benefit": "증분 편익",
        "cba_delta_net": "증분 순편익",
        "cba_bcr": "편익-비용 비 (BCR)",
        "cba_roi": "ROI",
        "cba_best": "순편익 최고: {name}",
        "cba_note_baseline": "기준",
        "cba_note_dominant": "더 저렴하고 열등하지 않음",
        "cba_note_dominated": "비용이 더 들고 추가 편익 없음",
        "cba_note_cheaper_less_benefit": "더 저렴, 편익 낮음",
        "cba_note_no_cost_change": "비용 변화 없음",
        "cba_baseline_info": "비교 기준: {name}",
    },
    "ja": {
        "analysis_type_cma": "費用最小化分析 (CMA)",
        "analysis_type_cba": "費用便益分析 (CBA)",
        "analysis_type_cma_help": "戦略間でアウトカムが同等の場合のみ使用：最も安い戦略を選びます。効果が異なる場合は DECIDE が警告します。",
        "analysis_type_cba_help": "すべて金額で評価：便益 = 効果1単位の金銭価値 x 効果（+ 追加便益）。そのうえで純便益 (NPV)、便益費用比 (BCR)、基準戦略に対する ROI を計算します。",
        "cma_tolerance": "効果を同等とみなす範囲 (%)",
        "cma_cheapest": "最も安い戦略：{name}",
        "cma_tie": "費用が同額：{names}",
        "cma_extra_cost": "最安値との差額",
        "cma_cheapest_flag": "最安",
        "cma_warning": "戦略間で効果が {spread:.1%} 異なります（許容範囲 {tol:.1%} を超過）。費用最小化はアウトカムが同等の場合のみ有効です -- CEA の使用を検討してください。",
        "cma_ok": "効果は {tol:.1%} 以内で同等です（差 {spread:.1%}）。費用最小化が適切です。",
        "cba_value_per_effect": "効果1単位の金銭価値",
        "cba_extra_benefits": "戦略ごとの追加金銭便益（割引済み・任意）",
        "cba_extra_benefit_for": "追加便益 -- {name}",
        "cba_discount_note": "費用と効果はモデルから取得し、モデルの割引率で割引済みのため、純便益は正味現在価値 (NPV) です。",
        "cba_benefit": "総便益",
        "cba_net_benefit": "純便益 (NPV)",
        "cba_delta_cost": "増分費用",
        "cba_delta_benefit": "増分便益",
        "cba_delta_net": "増分純便益",
        "cba_bcr": "便益費用比 (BCR)",
        "cba_roi": "ROI",
        "cba_best": "純便益が最大：{name}",
        "cba_note_baseline": "基準",
        "cba_note_dominant": "より安く劣らない",
        "cba_note_dominated": "費用が高く追加便益なし",
        "cba_note_cheaper_less_benefit": "より安いが便益は低い",
        "cba_note_no_cost_change": "費用変化なし",
        "cba_baseline_info": "比較基準：{name}",
    },
}


# --------------------------------------------------------------------------
# Report support (Word / Excel technical documentation reports)
# --------------------------------------------------------------------------
def report_section(
    settings: Mapping[str, object] | None,
    names: Sequence[str],
    costs: Sequence[float],
    effects: Sequence[float],
    maximize_effect: bool,
    baseline: str | None,
    cost_symbol: str = "$",
    cost_decimals: int = 2,
    effect_decimals: int = 3,
):
    """Table + explanatory notes for the report when the Analysis type is
    CMA or CBA; ``None`` otherwise.  The DataFrame carries ``attrs["title"]``
    and ``attrs["notes"]`` (list of paragraphs)."""

    import pandas as pd

    kind = (settings or {}).get("type")
    if kind not in ("CMA", "CBA") or not names:
        return None
    settings = settings or {}
    cd, ed = cost_decimals, effect_decimals
    if kind == "CMA":
        tol = float(settings.get("cma_tolerance_pct", 1.0) or 0.0) / 100.0
        res = cma_analysis(list(names), list(costs), list(effects), tolerance=tol)
        df = pd.DataFrame([
            {
                "Strategy": r.name,
                f"Expected cost ({cost_symbol})": round(r.cost, cd),
                "Expected effect": round(r.effect, ed),
                "Extra cost vs cheapest": round(r.extra_cost_vs_cheapest, cd),
                "Cheapest": "Yes" if r.is_cheapest else "",
            }
            for r in res.rows
        ])
        notes = [
            "Cost-minimization analysis (CMA) compares strategies on cost alone and is valid only when "
            "the strategies' health outcomes are equivalent.",
            (f"Effects are equivalent within the {tol:.1%} tolerance (observed spread {res.effect_spread:.1%}); "
             f"the cheapest strategy is {res.cheapest}.") if res.effects_equivalent else
            (f"WARNING: effects differ by {res.effect_spread:.1%} across strategies, more than the {tol:.1%} "
             f"tolerance, so the equivalence assumption is not supported. Cost-effectiveness analysis (ICER) "
             f"is the appropriate comparison; the cheapest strategy by cost alone is {res.cheapest}."),
        ]
        if res.ties:
            notes.append("Strategies tied on lowest cost: " + ", ".join([res.cheapest, *res.ties]) + ".")
        df.attrs["title"] = "Cost-minimization analysis"
        df.attrs["notes"] = notes
        return df

    value = float(settings.get("cba_value_per_effect", settings.get("willingness_to_pay", 0.0)) or 0.0)
    extra = dict(settings.get("cba_extra_benefits") or {})
    res = cba_analysis(list(names), list(costs), list(effects), value,
                       maximize_effect=maximize_effect, extra_benefits=extra, baseline=baseline)
    def r2(x):
        return "-" if x is None else round(x, cd)
    df = pd.DataFrame([
        {
            "Strategy": r.name,
            f"Cost ({cost_symbol})": round(r.cost, cd),
            "Effect": round(r.effect, ed),
            "Total benefit": round(r.benefit, cd),
            "Net benefit (NPV)": round(r.net_benefit, cd),
            "Incremental cost": r2(r.delta_cost),
            "Incremental benefit": r2(r.delta_benefit),
            "Incremental net benefit": r2(r.delta_net_benefit),
            "BCR": "-" if r.bcr is None else round(r.bcr, 2),
            "ROI": "-" if r.roi is None else f"{r.roi:.1%}",
            "Note": r.bcr_note.replace("_", " "),
        }
        for r in res.rows
    ])
    df.attrs["title"] = "Cost-benefit analysis"
    df.attrs["notes"] = [
        f"Cost-benefit analysis (CBA) expresses outcomes in money. Benefit = {value:,.{cd}f} per effect unit "
        f"x effect ({'maximized' if maximize_effect else 'minimized, so avoided events count as benefit'}) "
        "plus any extra monetary benefit entered per strategy. Net benefit = benefit - cost.",
        "Costs and effects come from the model and are already discounted with the model's discount rates, "
        "so net benefit is a net present value (NPV). Incremental figures, benefit-cost ratio (BCR = "
        f"incremental benefit / incremental cost) and ROI (incremental net benefit / incremental cost) are "
        f"relative to the baseline strategy ({res.baseline}).",
        f"The strategy with the highest net benefit is {res.best}.",
    ]
    return df


def write_report_section(doc, tables) -> None:
    """Append the CMA/CBA table to a report_docx.Docx (no-op if absent)."""

    df = tables.get("cma_cba")
    if df is None or df.empty:
        return
    doc.heading(df.attrs.get("title", "Additional analysis"), level=2)
    for note in df.attrs.get("notes", []):
        doc.paragraph(note)
    doc.table(df)
