import os, json, joblib, shutil, time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from tensorflow.keras import layers
import streamlit.components.v1 as components
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="Heavy Fleet Route Operations", layout="wide")

# --- UI State & Ledgers (Expanded for Full AI Control) ---
if 'fleet_state' not in st.session_state: st.session_state['fleet_state'] = {}
if 'maintenance_resets' not in st.session_state: st.session_state['maintenance_resets'] = {}
if 'selected_routes' not in st.session_state: st.session_state['selected_routes'] = []
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
if 'notif_key' not in st.session_state: st.session_state['notif_key'] = 0
if 'route_string_sidebar' not in st.session_state: st.session_state['route_string_sidebar'] = "60, 45, 80, 110"
if 'route_string_manager' not in st.session_state: st.session_state['route_string_manager'] = "60, 45, 80, 110"

# AI Specific Memory States
if 'ai_error' not in st.session_state: st.session_state['ai_error'] = None
if 'ai_harsh_routes' not in st.session_state: st.session_state['ai_harsh_routes'] = set()
if 'ai_etops_routes' not in st.session_state: st.session_state['ai_etops_routes'] = set()
if 'ai_priority_routes' not in st.session_state: st.session_state['ai_priority_routes'] = set()

# Economics & Operational Default States (Bound to UI later)
if 'econ_rev' not in st.session_state: st.session_state['econ_rev'] = 8000.0
if 'econ_op' not in st.session_state: st.session_state['econ_op'] = 1500.0
if 'econ_fail' not in st.session_state: st.session_state['econ_fail'] = 2500000.0
if 'econ_maint' not in st.session_state: st.session_state['econ_maint'] = 100000.0
if 'econ_unf' not in st.session_state: st.session_state['econ_unf'] = 50000.0
if 'op_det_buf' not in st.session_state: st.session_state['op_det_buf'] = 20
if 'op_via_prob' not in st.session_state: st.session_state['op_via_prob'] = 85
if 'harsh_env_val' not in st.session_state: st.session_state['harsh_env_val'] = 1.5
if 'prio_mult' not in st.session_state: st.session_state['prio_mult'] = 3.0

# --- AI Modular Function (Omnipotent Semantic Agent) ---
def run_ai_agent():
    """Isolated LLM logic with intent-based NLP and semantic understanding."""
    st.session_state['ai_error'] = None 
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        messages = st.session_state['notifications_df']['message'].tolist()
        if not messages: return
        
        prompt = f"""
        You are an advanced NLP aviation logistics parser. Read the following messages and extract the true operational intent. 
        You must understand the context and semantics—do not rely on exact phrasing.
        
        Messages to parse: {messages}
        
        Convert the intent of these messages into a JSON object using ONLY these keys:
        
        1. 'add_routes': [List of integers]. Extract new cycle lengths mentioned for creating routes (e.g. "opening a new route of 30 cycles" -> [30]).
        2. 'off_duty_engines': [List of integers]. Extract Engine/Unit IDs that are grounded, stopped, or called off duty for any reason (e.g. "Unit 1 was called off duty" -> [1]).
        3. 'send_to_maint': [List of integers]. Extract Engine/Unit IDs that broke down or explicitly need maintenance.
        4. 'complete_maint': [List of integers]. Extract Engine/Unit IDs that finished maintenance, are cleared, or ready to return to the viable pool (e.g. "Unit 3 maintenance complete. Asset is cleared" -> [3]).
        
        ROUTE-SPECIFIC FLAGS (If a message targets a specific route like "Route 3" or "Route_3", extract the integer 3):
        5. 'harsh_routes': [List of integers]. Routes with sandstorms, harsh weather, or qualitative recommendations to use a harsh multiplier (e.g. "Sandstorm on Route_3. Harsh multiplier recommended" -> [3]).
        6. 'etops_routes': [List of integers]. Routes with oceanic, water, or ETOPS warnings.
        7. 'priority_routes': [List of integers]. Routes marked as VIP, urgent, or high priority.
        8. 'dispatch_routes': [List of strings]. Exact route names specifically requested for today's dispatch (e.g. "we need to go on Route_2" -> ["Route_2"]).
        
        GLOBAL PARAMETERS:
        9. 'update_params': {{Dictionary}}. ONLY use this if a global numerical setting is explicitly changed to a specific number (e.g. "Change the global harsh multiplier to 2.0" -> {{"harsh_multiplier": 2.0}}). DO NOT put route-specific qualitative warnings here.
        
        Return ONLY valid JSON. If a category has no matches, return an empty list [].
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        plan = json.loads(response.choices[0].message.content)

        if plan.get('add_routes'):
            current_routes = st.session_state['route_string_sidebar']
            for r in plan['add_routes']: current_routes += f", {r}"
            st.session_state['route_string_sidebar'] = current_routes
            st.session_state['route_string_manager'] = current_routes
            
        if plan.get('off_duty_engines'):
            for eid in plan['off_duty_engines']:
                if int(eid) in st.session_state['fleet_state']: st.session_state['fleet_state'][int(eid)] = 'VIABLE'
        if plan.get('send_to_maint'):
            for eid in plan['send_to_maint']:
                if int(eid) in st.session_state['fleet_state']: st.session_state['fleet_state'][int(eid)] = 'IN_MAINTENANCE'
        if plan.get('complete_maint'):
            for eid in plan['complete_maint']:
                target_id = int(eid)
                if target_id in st.session_state['fleet_state']:
                    st.session_state['fleet_state'][target_id] = 'VIABLE'
                    db_path = st.session_state.get('db_path', 'master_db.csv')
                    if os.path.exists(db_path):
                        df = pd.read_csv(db_path)
                        u_df = df[df['unit'] == target_id]
                        if not u_df.empty: st.session_state['maintenance_resets'][target_id] = u_df['cycle'].max()

        if plan.get('harsh_routes'):
            for r_num in plan['harsh_routes']: st.session_state['ai_harsh_routes'].add(f"Route_{r_num}")
        if plan.get('etops_routes'):
            for r_num in plan['etops_routes']: st.session_state['ai_etops_routes'].add(f"Route_{r_num}")
        if plan.get('priority_routes'):
            for r_num in plan['priority_routes']: st.session_state['ai_priority_routes'].add(f"Route_{r_num}")

        if plan.get('dispatch_routes'):
            for r_name in plan['dispatch_routes']:
                if r_name not in st.session_state['selected_routes']:
                    st.session_state['selected_routes'].append(r_name)

        if plan.get('update_params'):
            p = plan['update_params']
            def safe_num(key, cast_type, current_val):
                if key in p:
                    try: return cast_type(p[key])
                    except (ValueError, TypeError): return current_val 
                return current_val

            st.session_state['econ_rev'] = safe_num('revenue', float, st.session_state['econ_rev'])
            st.session_state['econ_op'] = safe_num('op_cost', float, st.session_state['econ_op'])
            st.session_state['econ_fail'] = safe_num('fail_penalty', float, st.session_state['econ_fail'])
            st.session_state['econ_maint'] = safe_num('maint_cost', float, st.session_state['econ_maint'])
            st.session_state['econ_unf'] = safe_num('unfulfilled_cost', float, st.session_state['econ_unf'])
            st.session_state['op_det_buf'] = safe_num('determ_buffer', int, st.session_state['op_det_buf'])
            st.session_state['op_via_prob'] = safe_num('viable_prob', int, st.session_state['op_via_prob'])
            st.session_state['harsh_env_val'] = safe_num('harsh_multiplier', float, st.session_state['harsh_env_val'])
            st.session_state['prio_mult'] = safe_num('priority_multiplier', float, st.session_state['prio_mult'])
        
        st.session_state['notifications_df'] = pd.DataFrame(columns=['timestamp', 'from', 'message'])
        st.toast("AI Agent: Actions processed and applied!", icon="🤖")
        return True
        
    except Exception as e:
        st.session_state['ai_error'] = str(e)
        return False

# --- CSS DOM Hijack for Custom Styling ---
st.markdown("""
    <style>
        div[data-testid="stToast"] { background-color: #198754 !important; border-radius: 8px !important; border: 2px solid #146c43 !important; box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important; }
        div[data-testid="stToast"] * { color: #ffffff !important; font-weight: 600 !important; }
        div[data-testid="stPopoverBody"] { width: 380px !important; max-width: 95vw !important; padding: 12px !important; }
        div[data-testid="stPopoverBody"] p { margin-bottom: 0px !important; }
        div[data-testid="stPopoverBody"] button { padding: 0px 4px !important; font-size: 0.5rem !important; min-height: 22px !important; height: 22px !important; line-height: 1 !important; }
        div[data-testid="stPopoverBody"] div[data-testid="stButton"] { margin-bottom: -15px !important; }
    </style>
""", unsafe_allow_html=True)

# --- Audit Trail & Inbox Ledgers ---
if 'last_upload_time' not in st.session_state: st.session_state['last_upload_time'] = None
if 'last_assignment_time' not in st.session_state: st.session_state['last_assignment_time'] = None
if 'last_rollback_time' not in st.session_state: st.session_state['last_rollback_time'] = None
if 'notifications_df' not in st.session_state: 
    st.session_state['notifications_df'] = pd.DataFrame(columns=['timestamp', 'from', 'message'])

def sync_to_manager(): st.session_state['route_string_manager'] = st.session_state['route_string_sidebar']
def sync_to_sidebar(): st.session_state['route_string_sidebar'] = st.session_state['route_string_manager']
def clear_route_selection(): st.session_state['selected_routes'] = []

def commit_deployments(assign_df):
    for _, row in assign_df.iterrows():
        if row['Action'] == 'DEPLOY': 
            st.session_state['fleet_state'][int(row['Engine'])] = 'ON_DUTY'
    st.session_state['selected_routes'] = []
    st.session_state['last_assignment_time'] = pd.Timestamp.now(tz='Asia/Kolkata')
    st.toast("ASSETS DEPLOYED SUCCESSFULLY!", icon="🚀")

def execute_rollback(db_path):
    if os.path.exists(db_path + ".bak"):
        shutil.copy(db_path + ".bak", db_path)
        st.session_state['last_rollback_time'] = pd.Timestamp.now(tz='Asia/Kolkata')
        st.toast("DATABASE ROLLED BACK TO PREVIOUS STATE!", icon="⏪")

def process_alert(index, action_name):
    if index in st.session_state['notifications_df'].index:
        st.session_state['notifications_df'] = st.session_state['notifications_df'].drop(index).reset_index(drop=True)
    st.toast(f"Alert {action_name}!", icon="✔️")

# --- Custom Layers for Saved Model ---
class FFTLayer(layers.Layer):
    """Extracts Frequency Domain features using 1D FFT"""
    def call(self, inputs):
        x = tf.cast(inputs, tf.complex64)
        fft = tf.signal.fft(x)
        return tf.cast(tf.math.abs(fft), tf.float32)

class SEBlock(layers.Layer):
    def __init__(self, channels=32, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.avg = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(channels // ratio, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')
    def call(self, x): return x * tf.expand_dims(self.dense2(self.dense1(self.avg(x))), 1)
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'ratio': self.ratio})
        return config

RUL_CAP, WINDOW_SIZE = 125.0, 50

def get_survival_prob(q05, q50, q95, route_len): 
    return float(np.interp(route_len, [0.0, q05, q50, q95, q95 * 1.5], [1.0, 0.95, 0.50, 0.05, 0.0]))

def execute_fleet_logic(engines, master_routes, active_routes, econ, determ_buffer, viable_threshold, harsh_mult, prio_mult, mode='probabilistic'):
    viable_engines, assignments, total_profit = [], [], 0.0
    
    for e in engines:
        current_state = st.session_state['fleet_state'].get(e['id'], 'VIABLE')
        maint_ev = -((econ['maint'] / RUL_CAP) * e['q50']) if e['q50'] else 0.0
        
        if current_state not in ['VIABLE', 'REQUIRES_MAINTENANCE']:
            if current_state == 'IN_MAINTENANCE': total_profit += maint_ev
            continue

        is_viable = False
        for r in master_routes:
            eff_len = r['Length'] * harsh_mult if r['Harsh'] else r['Length']
            if r['ETOPS'] and e['q05'] < eff_len: continue 
            if mode == 'probabilistic' and get_survival_prob(e['q05'], e['q50'], e['q95'], eff_len) >= viable_threshold: is_viable = True; break
            elif mode != 'probabilistic' and e['q50'] >= eff_len + determ_buffer: is_viable = True; break

        if current_state == 'REQUIRES_MAINTENANCE': is_viable = False
        
        if is_viable: viable_engines.append(e)
        else: total_profit += maint_ev
            
    n_eng, n_route = len(viable_engines), len(active_routes)
    unfulfilled_count = 0
    
    if n_route > 0:
        cost_matrix = np.zeros((n_eng + n_route, n_eng + n_route))
        for i in range(n_eng + n_route):
            for j in range(n_eng + n_route):
                if i < n_eng and j < n_route: 
                    eff_len = active_routes[j]['Length'] * harsh_mult if active_routes[j]['Harsh'] else active_routes[j]['Length']
                    if active_routes[j]['ETOPS'] and viable_engines[i]['q05'] < eff_len: cost_matrix[i, j] = 1e9; continue
                    p_surv = get_survival_prob(viable_engines[i]['q05'], viable_engines[i]['q50'], viable_engines[i]['q95'], eff_len)
                    if mode == 'probabilistic':
                        cost_matrix[i, j] = -((p_surv * (econ['rev'] * active_routes[j]['Length'] - econ['op'] * active_routes[j]['Length'])) - ((1.0 - p_surv) * (econ['op'] * active_routes[j]['Length'] + econ['fail'])))
                    else:
                        cost_matrix[i, j] = -((econ['rev'] * active_routes[j]['Length']) - (econ['op'] * active_routes[j]['Length'])) if viable_engines[i]['q50'] >= eff_len + determ_buffer else econ['fail']
                elif i >= n_eng and j < n_route: 
                    cost_matrix[i, j] = econ['unfulfilled'] * prio_mult if active_routes[j]['Priority'] else econ['unfulfilled'] 
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        idle_engines = [viable_engines[i] for i, j in zip(row_ind, col_ind) if i < n_eng and j >= n_route]

        for i, j in zip(row_ind, col_ind):
            if i < n_eng and j < n_route: 
                e, r = viable_engines[i], active_routes[j]
                p_surv = get_survival_prob(e['q05'], e['q50'], e['q95'], r['Length'] * harsh_mult if r['Harsh'] else r['Length'])
                true_ev = (p_surv * (econ['rev'] * r['Length'] - econ['op'] * r['Length'])) - ((1.0 - p_surv) * (econ['op'] * r['Length'] + econ['fail']))
                total_profit += true_ev
                
                target_name = r['Route'] + (" 🌊" if r['ETOPS'] else "") + (" 🏜️" if r['Harsh'] else "") + (" ⭐" if r['Priority'] else "")
                assignments.append({'Route_Target': target_name, 'Engine': str(e['id']), 'Action': 'DEPLOY', 'P_Surv': p_surv, 'True_EV': true_ev})
            elif i >= n_eng and j < n_route: 
                r = active_routes[j]
                penalty = econ['unfulfilled'] * prio_mult if r['Priority'] else econ['unfulfilled']
                total_profit -= penalty
                unfulfilled_count += 1
                
                target_name = r['Route'] + (" 🌊" if r['ETOPS'] else "") + (" 🏜️" if r['Harsh'] else "") + (" ⭐" if r['Priority'] else "")
                assignments.append({'Route_Target': target_name, 'Engine': "INSUFFICIENT ENGINES" if not idle_engines else "ECONOMIC REASON", 'Action': 'UNFULFILLED', 'P_Surv': np.nan, 'True_EV': -penalty})
                
    assign_df = pd.DataFrame(assignments).sort_values('Route_Target') if assignments else pd.DataFrame(columns=['Route_Target', 'Engine', 'Action', 'P_Surv', 'True_EV'])
    return assign_df, total_profit, unfulfilled_count

# --- Sidebar UI ---
with st.sidebar:
    components.html("""
        <div style="font-family: 'Courier New', Courier, monospace; font-size: 1.1em; font-weight: bold; color: #ff4b4b; text-align: center; padding: 10px; border: 1px solid #ff4b4b; border-radius: 5px; background-color: #262730; margin-bottom: 10px; line-height: 1.4;">
            <div id="clock_date">Loading Date...</div><div id="clock_time">Loading Time...</div>
        </div>
        <script>
        function updateTime() { const now = new Date(); document.getElementById('clock_time').innerText = now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' }) + " IST"; document.getElementById('clock_date').innerText = now.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }); }
        setInterval(updateTime, 1000); updateTime();
        </script>
    """, height=80)

    events = []
    if st.session_state['last_upload_time']: events.append(("Data uploaded", st.session_state['last_upload_time']))
    if st.session_state['last_assignment_time']: events.append(("Fleet deployed", st.session_state['last_assignment_time']))
    if st.session_state['last_rollback_time']: events.append(("Database rollback", st.session_state['last_rollback_time']))
    
    if events:
        events.sort(key=lambda x: x[1], reverse=True)
        html_str = "<div style='text-align: center; font-size: 0.85em; color: #a0aec0; margin-top: -5px; margin-bottom: 20px; line-height: 1.6;'>"
        for label, ts in events:
            html_str += f"<div>{label} at <span style='color: #fff;'>{ts.strftime('%H:%M:%S, %d %b')}</span></div>"
        html_str += "</div>"
        st.markdown(html_str, unsafe_allow_html=True)

    st.header("Data Management")
    st.button("⏪ Undo Last Upload", use_container_width=True, on_click=execute_rollback, args=(st.session_state.get('db_path', 'master_db.csv'),), disabled=not os.path.exists(st.session_state.get('db_path', 'master_db.csv') + ".bak"))

    uploaded_file = st.file_uploader("Upload Flight Logs (CSV)", type="csv", key=f"up_{st.session_state['uploader_key']}")
    if uploaded_file:
        shutil.copy(st.session_state.db_path, st.session_state.db_path + ".bak")
        cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        new_df = pd.read_csv(uploaded_file, header=None, names=cols)
        
        returned_engines = new_df['unit'].unique()
        for eid in returned_engines:
            if st.session_state['fleet_state'].get(int(eid)) == 'ON_DUTY':
                st.session_state['fleet_state'][int(eid)] = 'VIABLE'

        pd.concat([pd.read_csv(st.session_state.db_path), new_df]).drop_duplicates(subset=['unit', 'cycle']).to_csv(st.session_state.db_path, index=False)
        st.session_state['last_upload_time'] = pd.Timestamp.now(tz='Asia/Kolkata')
        st.session_state['uploader_key'] += 1
        st.toast("FLIGHT LOGS APPENDED & DUTY ROSTER UPDATED!", icon="✅")
        st.rerun()

    st.header("All Possible Routes")
    st.text_input("Comma separated cycles", key="route_string_sidebar", on_change=sync_to_manager)

    with st.expander("Economic Parameters", expanded=False):
        rev_per_cycle = st.number_input("Revenue per Cycle ($)", step=1000.0, key="econ_rev")
        op_cost_per_cycle = st.number_input("Op Cost per Cycle ($)", step=500.0, key="econ_op")
        failure_penalty = st.number_input("Failure Penalty ($)", step=100000.0, key="econ_fail")
        maint_cost = st.number_input("Maintenance Cost ($)", step=10000.0, key="econ_maint")
        unfulfilled_cost = st.number_input("Unfulfillment Cost ($)", step=5000.0, key="econ_unf")
    econ_params = {'rev': rev_per_cycle, 'op': op_cost_per_cycle, 'fail': failure_penalty, 'maint': maint_cost, 'unfulfilled': unfulfilled_cost}
    
    with st.expander("Operational Rules", expanded=False):
        determ_buffer = st.number_input("Deterministic Safety Buffer", step=1, key="op_det_buf")
        viable_prob_check = st.slider("Viable Prob Check (%)", min_value=50, max_value=100, key="op_via_prob") / 100.0
        
    with st.expander("System Paths", expanded=False):
        model_path = st.text_input("Model File", "fd001_model.keras")
        scaler_path = st.text_input("Scaler File", "fd001_scaler.joblib")
        calib_path = st.text_input("Calibration File", "fd001_calibration.json")
        st.session_state.db_path = st.text_input("Master DB", "master_db.csv")
        
    st.markdown("---")
    st.header("📬 Alert Feed Upload")
    notif_file = st.file_uploader("Upload Messages (CSV)", type="csv", key=f"notif_{st.session_state['notif_key']}")
    if notif_file:
        try:
            new_notifs = pd.read_csv(notif_file)
            if all(c in new_notifs.columns for c in ['timestamp', 'from', 'message']):
                combined = pd.concat([st.session_state['notifications_df'], new_notifs])
                combined = combined.drop_duplicates(subset=['timestamp', 'message'])
                st.session_state['notifications_df'] = combined.sort_values('timestamp', ascending=False).reset_index(drop=True)
                st.session_state['notif_key'] += 1
                st.toast("NEW ALERTS RECEIVED!", icon="📬")
                st.rerun()
            else:
                st.error("CSV must contain 'timestamp', 'from', and 'message' columns.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(calib_path) and os.path.exists(st.session_state.db_path)): st.stop()

@st.cache_resource
def load_assets(m_path, s_path, c_path):
    m = tf.keras.models.load_model(m_path, custom_objects={'FFTLayer': FFTLayer, 'SEBlock': SEBlock}, compile=False)
    with open(c_path, 'r') as f: calib = json.load(f)
    return m, joblib.load(s_path), calib['features'], np.array(calib['baseline_state'])

model, scaler, feats, baseline_arr = load_assets(model_path, scaler_path, calib_path)
master_df = pd.read_csv(st.session_state.db_path)
X_input, engine_pool = [], []

for unit in master_df['unit'].unique():
    eid = int(unit)
    if eid not in st.session_state['fleet_state']: st.session_state['fleet_state'][eid] = 'VIABLE'
        
    u_df = master_df[master_df['unit'] == unit].sort_values('cycle')
    max_physical_cycle = u_df['cycle'].max()
    reset_cycle = st.session_state['maintenance_resets'].get(eid, 0)
    u_df = u_df[u_df['cycle'] > reset_cycle]
    
    active_cycles = len(u_df)
    engine_dict = {'id': eid, 'cycles': active_cycles, 'abs_max_cycle': max_physical_cycle}
    
    if active_cycles == 0: 
        engine_dict.update({'q05': RUL_CAP, 'q50': RUL_CAP, 'q95': RUL_CAP})
    else:
        # --- CRITICAL FIX FOR KEYERROR ---
        # If train.py saved integer indices, slice by location. If strings, slice by name.
        if isinstance(feats[0], int):
            feat_data = u_df.iloc[:, feats]
        else:
            feat_data = u_df[feats]
        # ---------------------------------
        
        scaled = scaler.transform(feat_data.tail(WINDOW_SIZE).values)
        padded = np.pad(scaled, ((WINDOW_SIZE - len(scaled), 0), (0, 0)), mode='edge') if len(scaled) < WINDOW_SIZE else scaled
        X_input.append(padded)
        
    engine_pool.append(engine_dict)

if X_input:
    preds = model.predict(np.array(X_input), verbose=0)
    idx = 0
    for e in engine_pool:
        if 'q50' not in e:
            e.update({'q05': (preds[0][idx][0] * RUL_CAP), 'q50': (preds[1][idx][0] * RUL_CAP), 'q95': (preds[2][idx][0] * RUL_CAP)})
            idx += 1

lengths = [int(x.strip()) for x in st.session_state['route_string_manager'].split(',') if x.strip().isdigit()]

grid = [
    {
        "Route": f"Route_{i+1}", 
        "Length": l, 
        "ETOPS": (f"Route_{i+1}" in st.session_state['ai_etops_routes']), 
        "Harsh": (f"Route_{i+1}" in st.session_state['ai_harsh_routes']), 
        "Priority": (f"Route_{i+1}" in st.session_state['ai_priority_routes'])
    } 
    for i, l in enumerate(lengths)
]

for e in engine_pool:
    if st.session_state['fleet_state'][e['id']] not in ['ON_DUTY', 'IN_MAINTENANCE']:
        is_viable = False
        for r in grid:
            eff_len = r['Length'] * st.session_state.get('harsh_env_val', 1.5) if r['Harsh'] else r['Length']
            if r['ETOPS'] and e['q05'] < eff_len: continue 
            if get_survival_prob(e['q05'], e['q50'], e['q95'], eff_len) >= viable_prob_check: 
                is_viable = True; break
        
        if not is_viable and st.session_state['fleet_state'][e['id']] == 'VIABLE':
            st.session_state['fleet_state'][e['id']] = 'REQUIRES_MAINTENANCE'

# --- Global Dashboard ---
header_col1, header_col2 = st.columns([8, 1])
with header_col1:
    st.title("Heavy Fleet Route Operations")
with header_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    num_alerts = len(st.session_state['notifications_df'])
    with st.popover(f"🔔 Inbox ({num_alerts})", use_container_width=True):
        if num_alerts == 0:
            st.write("All caught up! No new alerts.")
        else:
            st.button("🤖 Take action for all", type="primary", use_container_width=True, on_click=run_ai_agent)
            st.markdown("<hr style='margin: 8px 0px;'>", unsafe_allow_html=True)

            for idx, row in st.session_state['notifications_df'].head(5).iterrows():
                msg_hash = hash(str(row['timestamp']) + str(row['message']))
                st.markdown(f"<div style='font-size: 0.75rem; color: #a0aec0; margin-bottom: 2px;'><strong>{row['from']}</strong> &nbsp;•&nbsp; {row['timestamp']}</div>", unsafe_allow_html=True)
                
                nc1, nc2 = st.columns([2, 1])
                with nc1:
                    st.markdown(f"<div style='font-size: 0.99rem; line-height: 1.3;'>{row['message']}</div>", unsafe_allow_html=True)
                with nc2:
                    st.button("Done", key=f"act_{msg_hash}", on_click=process_alert, args=(idx, "marked as Done"), use_container_width=True)
                    st.button("Hold", key=f"hld_{msg_hash}", on_click=process_alert, args=(idx, "placed on Hold"), use_container_width=True)
                    st.button("Drop", key=f"ign_{msg_hash}", on_click=process_alert, args=(idx, "dropped"), use_container_width=True)
                
                st.markdown("<hr style='margin: 10px 0px; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
            
            if num_alerts > 5:
                st.caption(f"+ {num_alerts - 5} older messages hidden.")

if st.session_state.get('ai_error'):
    st.error(f"🚨 AI Agent encountered an error: {st.session_state['ai_error']}")
    error_msg = st.session_state['ai_error'].lower()
    if "api_key" in error_msg or "authentication" in error_msg:
        st.warning("Hint: Check your `.streamlit/secrets.toml` file. Ensure OPENAI_API_KEY is correct and wrapped in double quotes.")
    elif "insufficient_quota" in error_msg or "429" in error_msg:
        st.warning("Hint: Check your OpenAI billing dashboard. You may be out of API credits or hitting rate limits.")

# --- 🎯 Executive Fleet Overview (Donut Chart & Metrics) ---
state_counts = {'VIABLE': 0, 'ON_DUTY': 0, 'REQUIRES_MAINTENANCE': 0, 'IN_MAINTENANCE': 0}
for e in engine_pool:
    state_counts[st.session_state['fleet_state'][e['id']]] += 1

df_pie = pd.DataFrame({
    'Status': ['Viable (Standby)', 'On Duty', 'Req. Maintenance', 'In Maintenance'],
    'Count': [state_counts['VIABLE'], state_counts['ON_DUTY'], state_counts['REQUIRES_MAINTENANCE'], state_counts['IN_MAINTENANCE']],
    'Color': ['#198754', '#3296FF', '#dc3545', '#ffc107']
})
df_pie = df_pie[df_pie['Count'] > 0] 

col_chart, col_metrics = st.columns([1.5, 2.5])

with col_chart:
    if not df_pie.empty:
        fig = px.pie(df_pie, values='Count', names='Status', hole=0.45, color='Status', 
                     color_discrete_map={r['Status']: r['Color'] for _, r in df_pie.iterrows()})
        fig.update_traces(textposition='inside', textinfo='value+percent', hoverinfo='label+value')
        
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), 
            height=200, 
            showlegend=False, 
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>Total</b><br><b>{len(engine_pool)}</b>", x=0.5, y=0.5, font_size=18, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

with col_metrics:
    st.markdown("<br><br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Viable", state_counts['VIABLE'])
    m2.metric("🔵 On Duty", state_counts['ON_DUTY'])
    m3.metric("🔴 Req. Maint.", state_counts['REQUIRES_MAINTENANCE'])
    m4.metric("🟠 In Maint.", state_counts['IN_MAINTENANCE'])

st.markdown("---")
st.markdown("### Executive System Control")

with st.expander("🛠️ Mission Control Constraints", expanded=False):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: st.text_input("Master Route Pool", key="route_string_manager", on_change=sync_to_sidebar)
    with col2: harsh_multiplier = st.number_input("Harsh Environment", step=0.1, key="harsh_env_val")
    with col3: priority_multiplier = st.number_input("Priority VIP Penalty", step=0.5, key="prio_mult")

    master_route_configs = st.data_editor(
        pd.DataFrame(grid), 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Route": st.column_config.TextColumn("Target Name", disabled=True),
            "Length": st.column_config.NumberColumn("Cycles", disabled=True),
            "ETOPS": st.column_config.CheckboxColumn("🌊 ETOPS"),
            "Harsh": st.column_config.CheckboxColumn("🏜️ Harsh Env"),
            "Priority": st.column_config.CheckboxColumn("⭐ Priority")
        }
    ).to_dict('records')

compare_mode = st.checkbox("Compare with Legacy Deterministic Model", value=False)
tab1, tab2 = st.tabs(["1. Fleet Status", "2. Deployment Assignments"])

with tab2:
    st.markdown("### Executive Route Selector")
    st.multiselect("Select flights requiring dispatch today:", options=[r['Route'] for r in master_route_configs], key='selected_routes')
    active_route_configs = [r for r in master_route_configs if r['Route'] in st.session_state['selected_routes']]

prob_assign, prob_total, prob_unf = execute_fleet_logic(engine_pool, master_route_configs, active_route_configs, econ_params, determ_buffer, viable_prob_check, harsh_multiplier, priority_multiplier, mode='probabilistic')
det_assign, det_total, det_unf = execute_fleet_logic(engine_pool, master_route_configs, active_route_configs, econ_params, determ_buffer, viable_prob_check, harsh_multiplier, priority_multiplier, mode='deterministic')

def style_assign(df):
    if df.empty: return df
    return df.style.apply(lambda r: ['background-color: rgba(255, 165, 0, 0.2)']*len(r) if r['Action'] == 'UNFULFILLED' else ['']*len(r), axis=1).format({'P_Surv': '{:.4f}', 'True_EV': '${:,.2f}'}, na_rep='-')

with tab1:
    st.subheader("Current Fleet Health & Status")
    st.markdown("Asset records dynamically track active physical wear vs. predictions. Actions instantly trigger state transitions.")
    
    hc1, hc2, hc3, hc4, hc5, hc6, hc7 = st.columns([1.2, 1.5, 1.5, 1.5, 1.5, 2, 3])
    hc1.markdown("**Engine ID**"); hc2.markdown("**Cycles Logged**"); hc3.markdown("**Q05 Safe**"); hc4.markdown("**Q50 Median**"); hc5.markdown("**Q95 Limit**"); hc6.markdown("**Status**"); hc7.markdown("**Action**")
    st.markdown("---")
    
    for e in sorted(engine_pool, key=lambda x: x['id']):
        eid = e['id']
        state = st.session_state['fleet_state'][eid]
        
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.5, 1.5, 1.5, 1.5, 2, 3])
        c1.write(f"**Unit {eid}**"); c2.write(f"{e['cycles']}"); c3.write(f"{e['q05']:.1f}"); c4.write(f"{e['q50']:.1f}"); c5.write(f"{e['q95']:.1f}")
        
        if state == 'VIABLE': c6.success("VIABLE")
        elif state == 'ON_DUTY': c6.info("ON DUTY")
        elif state == 'REQUIRES_MAINTENANCE': c6.error("MAINTENANCE REQ")
        elif state == 'IN_MAINTENANCE': c6.warning("IN MAINTENANCE")
        
        with c7:
            if state == 'ON_DUTY':
                if st.button("📡 Recall Asset", key=f"rec_{eid}"): st.session_state['fleet_state'][eid] = 'VIABLE'; st.rerun()
            elif state in ['VIABLE', 'REQUIRES_MAINTENANCE']:
                if st.button("🔧 Send to Maint.", key=f"snd_{eid}"): st.session_state['fleet_state'][eid] = 'IN_MAINTENANCE'; st.rerun()
            elif state == 'IN_MAINTENANCE':
                if st.button("✅ Complete Maint.", key=f"rst_{eid}", type="primary"): st.session_state['fleet_state'][eid] = 'VIABLE'; st.session_state['maintenance_resets'][eid] = e['abs_max_cycle']; st.rerun()
                if st.button("⏪ Abort Maint.", key=f"abt_{eid}"): st.session_state['fleet_state'][eid] = 'VIABLE'; st.rerun()

with tab2:
    if not active_route_configs:
        st.info("Select routes above to view optimal asset deployment.")
    else:
        st.markdown("---")
        if compare_mode:
            col1, col2 = st.columns(2)
            with col1: 
                st.success(f"**Q-TFT Expected Profit:** ${prob_total:,.2f}")
                st.markdown(f"**Unfulfilled Routes:** {prob_unf}")
                if not prob_assign.empty: st.button("🚀 Commit Probabilistic Assignment", type="primary", on_click=commit_deployments, args=(prob_assign,))
            with col2: 
                st.error(f"**Legacy Expected Profit:** ${det_total:,.2f}")
                st.markdown(f"**Unfulfilled Routes:** {det_unf}")
                
            tc1, tc2 = st.columns(2)
            with tc1: st.markdown("**Probabilistic Matrix**"); st.dataframe(style_assign(prob_assign), use_container_width=True, hide_index=True)
            with tc2: st.markdown("**Deterministic Matrix**"); st.dataframe(style_assign(det_assign), use_container_width=True, hide_index=True)
        else:
            st.success(f"**Total Expected Profit:** ${prob_total:,.2f}")
            st.markdown(f"**Unfulfilled Routes:** {prob_unf}")
            if not prob_assign.empty: st.button("🚀 Commit Assignment", type="primary", on_click=commit_deployments, args=(prob_assign,))
            st.dataframe(style_assign(prob_assign), use_container_width=True, hide_index=True)
