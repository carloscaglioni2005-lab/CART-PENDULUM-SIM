import tempfile
from pathlib import Path

from solver import CircuitRequest


def _draw_with_schemdraw(req: CircuitRequest) -> Path | None:
    try:
        import schemdraw
        import schemdraw.elements as elm
    except Exception:
        return None

    if req.topology != "series" or not req.components:
        return None

    d = schemdraw.Drawing(show=False)

    for comp in req.components:
        kind = comp.kind.upper()
        label = f"{comp.label} {comp.value}".strip()

        if kind == "V":
            d += elm.SourceV().right().label(label)
        elif kind == "I":
            d += elm.SourceI().right().label(label)
        elif kind == "R":
            d += elm.Resistor().right().label(label)
        elif kind == "C":
            d += elm.Capacitor().right().label(label)
        elif kind == "L":
            d += elm.Inductor().right().label(label)
        elif kind == "GND":
            d += elm.Ground()
        else:
            d += elm.Line().right()

    d += elm.Line().down()
    d += elm.Line().left().length(max(1, len(req.components)))
    d += elm.Line().up()

    out = Path(tempfile.gettempdir()) / f"circuit_{abs(hash(str(req.components)))}.png"
    d.save(str(out))
    return out


def create_circuit_drawing(req: CircuitRequest) -> Path | None:
    if req.topology == "none":
        return None
    return _draw_with_schemdraw(req)
