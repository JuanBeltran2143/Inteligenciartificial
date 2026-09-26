import math
import heapq
import time
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# CAPÍTULO 2: LÓGICA Y REPRESENTACIÓN DEL CONOCIMIENTO
# ==============================================================================

class KnowledgeBase:
    """
    Base de Conocimiento (BC) que almacena la topología del sistema de transporte:
    - Estaciones con coordenadas relativas (x, y) en kilómetros.
    - Conexiones con servicios/rutas específicas y tiempos base.
    """
    def __init__(self):
        self.stations: Dict[str, Tuple[float, float]] = {}
        self.connections: List[Dict] = []
        self.rules: List['Rule'] = []

    def add_station(self, name: str, x: float, y: float):
        self.stations[name] = (x, y)

    def add_connection(self, station_a: str, station_b: str, line: str, travel_time: float):
        # Conexión bidireccional
        self.connections.append({'from': station_a, 'to': station_b, 'line': line, 'time': travel_time})
        self.connections.append({'from': station_b, 'to': station_a, 'line': line, 'time': travel_time})

    def add_rule(self, rule: 'Rule'):
        self.rules.append(rule)

    def get_neighbors(self, station: str) -> List[Dict]:
        return [c for c in self.connections if c['from'] == station]


# ==============================================================================
# CAPÍTULO 3: SISTEMA BASADO EN REGLAS (MOTOR DE INFERENCIA)
# ==============================================================================

class Rule:
    """Estructura de Regla Lógica: SI <condición> ENTONCES <acción>."""
    def __init__(self, name: str, condition_fn, action_fn):
        self.name = name
        self.condition_fn = condition_fn
        self.action_fn = action_fn


class RuleEngine:
    """Motor de Inferencia para evaluar transbordos y penalizaciones."""
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def evaluate_transition(self, current_line: Optional[str], next_line: str, base_time: float) -> Tuple[bool, float, List[str]]:
        context = {
            'previous_line': current_line,
            'next_line': next_line,
            'cost': base_time,
            'is_valid': True,
            'applied_rules': []
        }

        for rule in self.kb.rules:
            if rule.condition_fn(context):
                rule.action_fn(context)
                context['applied_rules'].append(rule.name)

        return context['is_valid'], context['cost'], context['applied_rules']


# ==============================================================================
# CAPÍTULO 9: BÚSQUEDA HEURÍSTICA (ALGORITMO A*)
# ==============================================================================

def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Heurística h(n): Estimación en línea recta entre coordenadas."""
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    return math.sqrt(dx*dx + dy*dy)


class MassTransitAStar:
    """Agente de búsqueda inteligente que calcula la mejor ruta usando A*."""
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.rule_engine = RuleEngine(kb)

    def search_best_route(self, start: str, goal: str):
        if start not in self.kb.stations or goal not in self.kb.stations:
            return {'status': 'INVALID_STATION'}

        goal_coords = self.kb.stations[goal]
        open_set = []
        counter = 0

        h_start = euclidean_distance(self.kb.stations[start], goal_coords)
        heapq.heappush(open_set, (h_start, counter, start, None, [(start, None, 0.0, "Origen")], 0.0))

        visited = {}

        while open_set:
            f_score, _, current_station, current_line, path, g_score = heapq.heappop(open_set)

            if current_station == goal:
                return {
                    'path': path,
                    'total_time': g_score,
                    'status': 'SUCCESS'
                }

            state_key = (current_station, current_line)
            if state_key in visited and visited[state_key] <= g_score:
                continue
            visited[state_key] = g_score

            for conn in self.kb.get_neighbors(current_station):
                neighbor = conn['to']
                next_line = conn['line']
                base_time = conn['time']

                is_valid, transition_cost, applied_rules = self.rule_engine.evaluate_transition(
                    current_line, next_line, base_time
                )

                if not is_valid:
                    continue

                new_g = g_score + transition_cost
                h_score = euclidean_distance(self.kb.stations[neighbor], goal_coords)
                new_f = new_g + h_score

                rule_info = f"Aplicó: {', '.join(applied_rules)}" if applied_rules else "Tramo directo"
                new_path = path + [(neighbor, next_line, transition_cost, rule_info)]

                counter += 1
                heapq.heappush(open_set, (new_f, counter, neighbor, next_line, new_path, new_g))

        return {'status': 'NO_ROUTE_FOUND'}


# ==============================================================================
# CONFIGURACIÓN DEL SISTEMA Y BASE DE CONOCIMIENTO
# ==============================================================================

def setup_system() -> KnowledgeBase:
    kb = KnowledgeBase()

    # 1. Estaciones y Coordenadas Geográficas (x, y)
    kb.add_station("Distrito Graffiti", 2.0, 1.0)
    kb.add_station("Ricaurte", 1.0, 3.0)
    kb.add_station("Calle 75 - Zona M", 2.0, 6.0)
    kb.add_station("Héroes", 2.0, 7.5)
    kb.add_station("Portal Norte", 2.0, 11.0)

    # 2. Conexiones Directas y Rutas/Servicios
    # Ruta E32: Conecta directo Distrito Graffiti con Calle 75 - Zona M
    kb.add_connection("Distrito Graffiti", "Calle 75 - Zona M", "Ruta E32", 18.0)

    # Ruta B12: Conecta Calle 75 - Zona M con Portal Norte
    kb.add_connection("Calle 75 - Zona M", "Portal Norte", "Ruta B12", 22.0)

    # Alternativa con múltiples transbordos (para probar la penalización)
    kb.add_connection("Distrito Graffiti", "Ricaurte", "Ruta F28", 8.0)
    kb.add_connection("Ricaurte", "Héroes", "Ruta B11", 17.0)
    kb.add_connection("Héroes", "Portal Norte", "Ruta B74", 12.0)

    # 3. Reglas Lógicas del Sistema
    def cond_transbordo(ctx):
        # Hay transbordo si venimos de una ruta previa y cambiamos a una distinta
        return ctx['previous_line'] is not None and ctx['previous_line'] != ctx['next_line']

    def act_transbordo(ctx):
        # Penalización de 5 minutos por transbordo
        ctx['cost'] += 5.0

    kb.add_rule(Rule("Penalización por Transbordo (+5 min)", cond_transbordo, act_transbordo))

    return kb


# ==============================================================================
# INTERFAZ DE USUARIO E INTERACCIÓN
# ==============================================================================

def main():
    kb = setup_system()
    agent = MassTransitAStar(kb)

    print("==========================================================")
    print("      SISTEMA INTELIGENTE DE RUTA ÓPTIMA - TRANSMILENIO   ")
    print("==========================================================")
    print("Estaciones disponibles en el sistema:")
    for station in kb.stations.keys():
        print(f" - {station}")
    print("----------------------------------------------------------")

    # Lectores de entrada
    origen_in = input("\nIngrese la estación de ORIGEN: ").strip()
    destino_in = input("Ingrese la estación de DESTINO: ").strip()

    # Normalización simple para coincidir con la Base de Conocimiento
    origen = next((s for s in kb.stations if s.lower() == origen_in.lower()), origen_in)
    destino = next((s for s in kb.stations if s.lower() == destino_in.lower()), destino_in)

    print("\n[...] Analizando ruta óptima y evaluando reglas de transbordo...")
    time.sleep(1.2)  # Simulación de cómputo del agente inteligente

    result = agent.search_best_route(origen, destino)

    print("\n==========================================================")
    print("                 RESULTADOS DE LA BÚSQUEDA                ")
    print("==========================================================")

    if result['status'] == 'SUCCESS':
        print(f"Origen:  {origen}")
        print(f"Destino: {destino}")
        print(f"Estimación de tiempo total: {result['total_time']:.2f} minutos")
        print("----------------------------------------------------------\n")
        print("Desglose paso a paso de la mejor alternativa:")
        
        for i, step in enumerate(result['path']):
            station, line, cost, info = step
            if i == 0:
                print(f"  1. Iniciar recorrido en la estación: [{station}]")
            else:
                print(f"  {i+1}. Tomar [{line}] hasta [{station}] | Tiempo tramo: {cost:.2f} min ({info})")
                
        print("\nNota: La ruta incluye la penalización de 5 min por cada transbordo de servicio.")
    elif result['status'] == 'INVALID_STATION':
        print(" Error: Una o ambas estaciones ingresadas no existen en la Base de Conocimiento.")
    else:
        print(" No se encontró una ruta válida entre los puntos indicados.")


if __name__ == "__main__":
    main()