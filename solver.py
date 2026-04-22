import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class PlotRequest:
    title: str
    x_label: str
    y_label: str
    x_min: float
    x_max: float
    expression: str


@dataclass
class CircuitComponent:
    kind: str
    label: str
    value: str


@dataclass
class CircuitRequest:
    topology: str = "none"
    components: list[CircuitComponent] = field(default_factory=list)
    instructions: str = ""


@dataclass
class SolveResult:
    title: str
    problem_rewrite: str
    given_data: list[str]
    steps: list[str]
    equations: list[str]
    final_answer: str
    sanity_checks: list[str]
    plot_requests: list[PlotRequest] = field(default_factory=list)
    circuit_request: CircuitRequest = field(default_factory=CircuitRequest)
    notes: str = ""


SYSTEM_PROMPT = """
Sei un tutor di elettronica per studenti principianti.
Devi leggere una foto di esercizio e rispondere SEMPRE in JSON valido.

Linee guida:
- Spiegazione chiara, in italiano.
- Evita salti logici.
- Se dati mancanti o illeggibili, dichiaralo in notes.
- Non inventare informazioni: fai assunzioni esplicite.
- Se utile, proponi grafici matematici con expression in funzione di x.
- Per circuiti semplici in serie, compila circuit_request con topology='series' e lista componenti ordinata.
- Tipi componenti ammessi: V, I, R, C, L, GND.

JSON richiesto:
{
  "title": string,
  "problem_rewrite": string,
  "given_data": string[],
  "steps": string[],
  "equations": string[],
  "final_answer": string,
  "sanity_checks": string[],
  "plot_requests": [
    {
      "title": string,
      "x_label": string,
      "y_label": string,
      "x_min": number,
      "x_max": number,
      "expression": string
    }
  ],
  "circuit_request": {
    "topology": "none" | "series",
    "components": [
      {"kind": "V|I|R|C|L|GND", "label": string, "value": string}
    ],
    "instructions": string
  },
  "notes": string
}
""".strip()


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_result(raw: dict[str, Any]) -> SolveResult:
    plot_requests: list[PlotRequest] = []
    for p in raw.get("plot_requests", []) or []:
        plot_requests.append(
            PlotRequest(
                title=str(p.get("title", "Grafico")),
                x_label=str(p.get("x_label", "x")),
                y_label=str(p.get("y_label", "y")),
                x_min=_safe_float(p.get("x_min"), -10.0),
                x_max=_safe_float(p.get("x_max"), 10.0),
                expression=str(p.get("expression", "x")),
            )
        )

    c = raw.get("circuit_request", {}) or {}
    components: list[CircuitComponent] = []
    for item in c.get("components", []) or []:
        components.append(
            CircuitComponent(
                kind=str(item.get("kind", "")).upper(),
                label=str(item.get("label", "")),
                value=str(item.get("value", "")),
            )
        )

    circuit_request = CircuitRequest(
        topology=str(c.get("topology", "none")).lower(),
        components=components,
        instructions=str(c.get("instructions", "")),
    )

    return SolveResult(
        title=str(raw.get("title", "Esercizio di elettronica")),
        problem_rewrite=str(raw.get("problem_rewrite", "")),
        given_data=[str(x) for x in (raw.get("given_data", []) or [])],
        steps=[str(x) for x in (raw.get("steps", []) or [])],
        equations=[str(x) for x in (raw.get("equations", []) or [])],
        final_answer=str(raw.get("final_answer", "")),
        sanity_checks=[str(x) for x in (raw.get("sanity_checks", []) or [])],
        plot_requests=plot_requests,
        circuit_request=circuit_request,
        notes=str(raw.get("notes", "")),
    )


def solve_from_image(
    image_bytes: bytes,
    mime_type: str,
    model_name: str,
    explanation_level: str,
) -> SolveResult:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non trovata. Inseriscila in .env o nelle variabili ambiente.")

    client = OpenAI(api_key=api_key)

    data_url = _to_data_url(image_bytes, mime_type)

    user_prompt = (
        f"Risolvi l'esercizio in modo {explanation_level}. "
        "Se nella foto ci sono piu esercizi, risolvi il primo completo e segnala la scelta in notes."
    )

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return _parse_result(parsed)
