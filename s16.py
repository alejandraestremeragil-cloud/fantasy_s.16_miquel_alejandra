import argparse
import heapq
import math
import random
import timeit
from dataclasses import dataclass
from itertools import combinations, product
from datos import JUGADORES, CUPOS, PRESUPUESTO

EPS = 1e-9


@dataclass
class Nodo:
    indice: int
    coste: int
    puntos: float
    cantidades: tuple
    elegidos: tuple


def coste_equipo(equipo):
    return sum(j["precio"] for j in equipo)


def puntos_equipo(equipo):
    return sum(j["puntos"] for j in equipo)


def es_equipo_valido(equipo, presupuesto, cupos):
    return coste_equipo(equipo) <= presupuesto and all(
        sum(j["posicion"] == pos for j in equipo) == cantidad
        for pos, cantidad in cupos.items()
    )


def voraz(jugadores, presupuesto, cupos, criterio):
    """Fichar por puntos o puntos/precio sin deshacer decisiones anteriores."""
    if criterio not in ("puntos", "ratio"):
        raise ValueError("Criterio: puntos o ratio")
    clave = (lambda j: j["puntos"]) if criterio == "puntos" else (
        lambda j: j["puntos"] / j["precio"]
    )
    equipo, coste = [], 0
    cantidades = dict.fromkeys(cupos, 0)
    for j in sorted(jugadores, key=clave, reverse=True):
        pos = j["posicion"]
        if cantidades[pos] < cupos[pos] and coste + j["precio"] <= presupuesto:
            equipo.append(j)
            coste += j["precio"]
            cantidades[pos] += 1
    return equipo


def contar_combinaciones(jugadores, cupos):
    return math.prod(math.comb(sum(j["posicion"] == pos for j in jugadores), n)
                     for pos, n in cupos.items())


def fuerza_bruta(jugadores, presupuesto, cupos, max_segundos=None):
    """Todas las combinaciones por posición; admite una parada de seguridad."""
    grupos = [[j for j in jugadores if j["posicion"] == pos] for pos in cupos]
    total = contar_combinaciones(jugadores, cupos)
    opciones = [combinations(grupo, n) for grupo, n in zip(grupos, cupos.values())]
    mejor, mejor_puntuacion = [], float("-inf")
    probados = viables = 0
    inicio = timeit.default_timer()
    terminado = True

    for seleccion in product(*opciones):
        if max_segundos is not None and probados % 1024 == 0:
            if timeit.default_timer() - inicio >= max_segundos:
                terminado = False
                break
        equipo = [j for grupo in seleccion for j in grupo]
        probados += 1
        if coste_equipo(equipo) > presupuesto:
            continue
        viables += 1
        puntos = puntos_equipo(equipo)
        if puntos > mejor_puntuacion + EPS:
            mejor, mejor_puntuacion = equipo, puntos

    return {"equipo": mejor, "puntos": mejor_puntuacion if mejor else None,
            "probados": probados, "viables": viables,
            "total": total, "terminado": terminado}


def cota_fraccionaria_sin_cupos(jugadores, dinero):
    """Mochila fraccionaria sin límites por posición (cota superior)."""
    puntos = 0.0
    for j in jugadores:  # Ordenados por puntos/precio
        fraccion = min(1, max(0, dinero) / j["precio"])
        puntos += fraccion * j["puntos"]
        dinero -= fraccion * j["precio"]
        if dinero <= EPS:
            break
    return puntos


def cota_superior(nodo, jugadores, presupuesto, cupos, posiciones):
    """Mínimo de dos cotas seguras: por puestos y mochila fraccionaria."""
    if nodo.coste > presupuesto:
        return float("-inf")
    pendientes = jugadores[nodo.indice:]
    coste_minimo, cota_puestos = nodo.coste, nodo.puntos
    for i, pos in enumerate(posiciones):
        faltan = cupos[pos] - nodo.cantidades[i]
        disponibles = [j for j in pendientes if j["posicion"] == pos]
        if faltan < 0 or faltan > len(disponibles):
            return float("-inf")
        coste_minimo += sum(sorted(j["precio"] for j in disponibles)[:faltan])
        cota_puestos += sum(sorted((j["puntos"] for j in disponibles),
                                  reverse=True)[:faltan])
    if coste_minimo > presupuesto:
        return float("-inf")
    cota_mochila = nodo.puntos + cota_fraccionaria_sin_cupos(
        pendientes, presupuesto - nodo.coste)
    return min(cota_puestos, cota_mochila)


def cota_fraccionaria_ingenua_con_cupos(jugadores, presupuesto, cupos):
    """Trampa del enunciado: NO es una cota superior válida."""
    faltan = dict(cupos)
    puntos, dinero = 0.0, presupuesto
    orden = sorted(jugadores, key=lambda j: j["puntos"] / j["precio"], reverse=True)
    for j in orden:
        fraccion = min(1, faltan[j["posicion"]], max(0, dinero) / j["precio"])
        puntos += fraccion * j["puntos"]
        dinero -= fraccion * j["precio"]
        faltan[j["posicion"]] -= fraccion
    return puntos


def solucion_inicial(jugadores, presupuesto, cupos):
    """Un once factible para empezar a podar, si existe."""
    baratos = []
    for pos, n in cupos.items():
        grupo = sorted((j for j in jugadores if j["posicion"] == pos),
                       key=lambda j: j["precio"])
        baratos.extend(grupo[:n])
    if not es_equipo_valido(baratos, presupuesto, cupos):
        return []
    candidatos = [baratos, voraz(jugadores, presupuesto, cupos, "puntos"),
                  voraz(jugadores, presupuesto, cupos, "ratio")]
    return max((e for e in candidatos if es_equipo_valido(e, presupuesto, cupos)),
               key=puntos_equipo)


def branch_and_bound(jugadores, presupuesto, cupos):
    """Decisiones fichar/no fichar; best-first con heapq y poda por cota."""
    posiciones = tuple(cupos)
    orden = sorted(jugadores, key=lambda j: j["puntos"] / j["precio"], reverse=True)
    mejor = solucion_inicial(orden, presupuesto, cupos)
    max_puntos = puntos_equipo(mejor) if mejor else float("-inf")
    raiz = Nodo(0, 0, 0.0, (0,) * len(cupos), ())
    limite = cota_superior(raiz, orden, presupuesto, cupos, posiciones)
    if limite == float("-inf"):
        return {"equipo": [], "puntos": None, "abiertos": 0, "generados": 0, "podados": 1}

    cola = [(-limite, 0, raiz)]  # Negativo: heapq es un heap de mínimos
    contador = abiertos = podados = 0
    generados = 1
    while cola:
        menos_limite, _, nodo = heapq.heappop(cola)
        abiertos += 1
        if -menos_limite <= max_puntos + EPS:
            podados += 1
            continue
        if all(nodo.cantidades[i] == cupos[pos] for i, pos in enumerate(posiciones)):
            max_puntos = nodo.puntos
            mejor = [orden[i] for i in nodo.elegidos]
            continue
        if nodo.indice == len(orden):
            podados += 1
            continue

        j = orden[nodo.indice]
        pos = posiciones.index(j["posicion"])
        # Dos ramas: fichar el jugador actual o descartarlo.
        for fichar in (True, False):
            if fichar and (nodo.cantidades[pos] >= cupos[j["posicion"]]
                            or nodo.coste + j["precio"] > presupuesto):
                podados += 1
                continue
            cantidades = list(nodo.cantidades)
            if fichar:
                cantidades[pos] += 1
            hijo = Nodo(nodo.indice + 1,
                        nodo.coste + (j["precio"] if fichar else 0),
                        nodo.puntos + (j["puntos"] if fichar else 0),
                        tuple(cantidades),
                        nodo.elegidos + (nodo.indice,) if fichar else nodo.elegidos)
            limite = cota_superior(hijo, orden, presupuesto, cupos, posiciones)
            if limite > max_puntos + EPS:
                contador += 1  # Evita comparar objetos Nodo en empates
                heapq.heappush(cola, (-limite, contador, hijo))
                generados += 1
            else:
                podados += 1

    return {"equipo": mejor, "puntos": max_puntos if mejor else None,
            "abiertos": abiertos, "generados": generados, "podados": podados}


def generar_jugadores_aleatorios(semilla=16):
    azar = random.Random(semilla)
    jugadores = []
    for pos, cantidad in {"POR": 8, "DEF": 21, "MED": 21, "DEL": 10}.items():
        for i in range(1, cantidad + 1):
            jugadores.append({"nombre": f"{pos}_{i:02d}", "posicion": pos,
                              "precio": azar.randint(3, 24),
                              "puntos": round(azar.uniform(3, 10), 1)})
    return jugadores


def medir(funcion, *args, **kwargs):
    resultado = None

    def ejecutar():
        nonlocal resultado
        resultado = funcion(*args, **kwargs)

    segundos = timeit.timeit(ejecutar, number=1)
    return resultado, segundos


def imprimir_equipo(nombre, equipo, presupuesto, cupos):
    print(f"\n{nombre}: {len(equipo)}/{sum(cupos.values())} jugadores | "
          f"{coste_equipo(equipo)}/{presupuesto} M€ | "
          f"{puntos_equipo(equipo):.1f} puntos | "
          f"Válido: {'sí' if es_equipo_valido(equipo, presupuesto, cupos) else 'no'}")
    for pos in cupos:
        print(f"  {pos}: {', '.join(j['nombre'] for j in equipo if j['posicion'] == pos) or '—'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-escala", action="store_true")
    parser.add_argument("--limite-bruta", type=float, default=1.0)
    args = parser.parse_args()

    formacion = "-".join(str(n) for n in CUPOS.values())
    print(f"PRÁCTICA S16 - FANTASY ({PRESUPUESTO} M€, {formacion})")
    for criterio in ("puntos", "ratio"):
        equipo = voraz(JUGADORES, PRESUPUESTO, CUPOS, criterio)
        imprimir_equipo(f"1. Voraz por {criterio}", equipo, PRESUPUESTO, CUPOS)

    bruto, t_bruto = medir(fuerza_bruta, JUGADORES, PRESUPUESTO, CUPOS)
    bnb, t_bnb = medir(branch_and_bound, JUGADORES, PRESUPUESTO, CUPOS)
    imprimir_equipo("2. Fuerza bruta", bruto["equipo"], PRESUPUESTO, CUPOS)
    imprimir_equipo("3. Branch and Bound", bnb["equipo"], PRESUPUESTO, CUPOS)
    print(f"Fuerza bruta: {bruto['probados']:,} equipos, {bruto['viables']:,} viables, {t_bruto:.4f} s")
    print(f"Branch and Bound: {bnb['abiertos']:,} nodos abiertos, "
          f"{bnb['podados']:,} podados, {t_bnb:.4f} s")
    assert bruto["puntos"] is not None and bnb["puntos"] is not None
    assert abs(bruto["puntos"] - bnb["puntos"]) < EPS

    orden = sorted(JUGADORES, key=lambda j: j["puntos"] / j["precio"], reverse=True)
    raiz = Nodo(0, 0, 0.0, (0,) * len(CUPOS), ())
    print("\n4. Cotas: ingenua =", round(cota_fraccionaria_ingenua_con_cupos(
        JUGADORES, PRESUPUESTO, CUPOS), 1), "; válida =",
          round(cota_superior(raiz, orden, PRESUPUESTO, CUPOS, tuple(CUPOS)), 1))

    if not args.sin_escala:
        jugadores = generar_jugadores_aleatorios()
        bruto60, t_bruto60 = medir(fuerza_bruta, jugadores, PRESUPUESTO, CUPOS,
                                  max_segundos=args.limite_bruta)
        bnb60, t_bnb60 = medir(branch_and_bound, jugadores, PRESUPUESTO, CUPOS)
        print(f"\n5. Escala: 60 jugadores, {contar_combinaciones(jugadores, CUPOS):,} equipos posibles")
        print(f"Fuerza bruta: {bruto60['probados']:,} probados, {t_bruto60:.3f} s; "
              f"{'terminada' if bruto60['terminado'] else 'interrumpida (sin óptimo certificado)'}")
        print(f"Branch and Bound: {bnb60['puntos']:.1f} puntos, "
              f"{bnb60['abiertos']} nodos, {t_bnb60:.4f} s")
        if bruto60["terminado"]:
            assert abs(bruto60["puntos"] - bnb60["puntos"]) < EPS


if __name__ == "__main__":
    main()
