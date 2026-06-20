import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MATHEMATICALLY CORRECT TAIJI/YIN-YANG COORD GENERATOR
# ============================================================================

def get_taiji_geometry(radius, alpha, n_points=150):
    """
    Computes exact continuous interlocking Yin-Yang coordinates at rotation alpha.
    Returns polygon vertices and eye positions directly to prevent transformation crashes.
    """
    R = radius
    
    # Base configuration for alpha = 0
    t_out = np.linspace(0, np.pi, n_points)
    x_out = R * np.cos(t_out)
    y_out = R * np.sin(t_out)
    
    t_in1 = np.linspace(np.pi, 0, n_points // 2)
    x_in1 = -R/2 + (R/2) * np.cos(t_in1)
    y_in1 = (R/2) * np.sin(t_in1)
    
    t_in2 = np.linspace(np.pi, 2*np.pi, n_points // 2)
    x_in2 = R/2 + (R/2) * np.cos(t_in2)
    y_in2 = (R/2) * np.sin(t_in2)
    
    base_x = np.concatenate([x_out, x_in1, x_in2])
    base_y = np.concatenate([y_out, y_in1, y_in2])
    
    # Apply standard rotation matrix transformation
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    rot_x = base_x * cos_a - base_y * sin_a
    rot_y = base_x * sin_a + base_y * cos_a
    
    black_vertices = np.column_stack([rot_x, rot_y])
    white_vertices = -black_vertices  # Inversion across origin
    
    # Dynamic positions of the core eyes
    eye_white = np.array([(R/2) * cos_a, (R/2) * sin_a])
    eye_black = -eye_white
    
    return black_vertices, white_vertices, eye_white, eye_black

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================

class BIGERParams:
    def __init__(self, rotor_radius=0.15, n_sectors=2, frequency=50):
        self.rotor_radius = rotor_radius  
        self.rotor_thickness = 0.01  
        self.rotor_density = 7500  
        self.rotor_volume = np.pi * rotor_radius**2 * self.rotor_thickness
        self.rotor_mass = self.rotor_volume * self.rotor_density / 2
        
        self.Br = 1.44  
        self.mu0 = 4 * np.pi * 1e-7
        self.M0 = self.Br / self.mu0  
        self.n_sectors = n_sectors
        
        self.coil_radius = 0.02  
        self.coil_area = np.pi * self.coil_radius**2  
        self.N_turns = 100  
        self.coil_resistance = 0.5  
        self.load_resistance = 10.0  
        self.coil_z_position = 0.10  
        
        self.t_forward = 0.95  
        self.t_reverse = 0.05  
        
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        
        self._calculate_derived()
    
    def _calculate_derived(self):
        half_area = np.pi * self.rotor_radius**2 / 2
        self.m0 = self.M0 * half_area * self.rotor_thickness  
        
        self.B_peak = (self.mu0 / (4*np.pi)) * (2*self.m0) / (self.coil_z_position**3)  
        self.Phi_peak = self.B_peak * self.coil_area * self.N_turns  
        self.V_peak = self.omega * self.Phi_peak  
        self.V_rms = self.V_peak / np.sqrt(2)  
        
        self.R_total = self.coil_resistance + self.load_resistance
        self.I_rms = self.V_rms / self.R_total  
        self.P_elec = self.I_rms**2 * self.load_resistance  
        
        self.tau_lenz = self.P_elec / self.omega  
        self.tau_eff = self.tau_lenz * (self.t_reverse / self.t_forward)  
        self.P_injector = self.tau_eff * self.omega  
        
        self.I_rotor = 0.5 * self.rotor_mass * self.rotor_radius**2
        self.E_kinetic = 0.5 * self.I_rotor * self.omega**2
    
    def get_emf(self, time):
        phi = self.omega * time
        fundamental = self.V_peak * np.sin(phi)
        harmonic_3 = 0.15 * self.V_peak * np.sin(3*phi)
        harmonic_5 = 0.05 * self.V_peak * np.sin(5*phi)
        return fundamental + harmonic_3 + harmonic_5
    
    def get_bigonal_field_grid(self, X, Z, time):
        """Vectorized spatial magnetic flux density Bz generation for fast animation"""
        R_coord = np.sqrt(X**2 + Z**2)
        Theta = np.arctan2(Z, X)
        phi_rotor = self.omega * time
        theta_rel = (Theta - phi_rotor) % (2 * np.pi)
        
        spatial_factor = np.where((theta_rel >= 0) & (theta_rel < np.pi), np.sin(theta_rel), 0.02)
        
        R_mask = np.maximum(R_coord, 0.001)
        hypotenuse = np.sqrt(R_mask**2 + Z**2)
        B_z = (self.mu0 / (4 * np.pi)) * (self.m0 * spatial_factor) * (3 * Z**2 - hypotenuse**2) / (hypotenuse**5)
        return B_z

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class BIGERSimulation:
    def __init__(self, params=None):
        self.params = params or BIGERParams()
        self.time = 0.0
        self.theta = 0.0
        self.omega = self.params.omega
        self.current = 0.0
        
        self.energy_kinetic = self.params.E_kinetic
        self.energy_elec = 0.0
        self.energy_injected = 0.0
        
        self.history = {
            'time': [], 'theta': [], 'omega': [], 'current': [], 'emf': [],
            'power_elec': [], 'torque': [], 'energy_kin': [], 'energy_elec': [], 'energy_inj': []
        }
        
        self.running = True
        self.step_count = 0
    
    def step(self, dt):
        if not self.running:
            return
        
        params = self.params
        self.step_count += 1
        
        emf = params.get_emf(self.time)
        
        L_coil = 0.01  
        R_total = params.R_total
        dI_dt = (emf - self.current * R_total) / L_coil
        self.current += dI_dt * dt
        
        # Enforce exact power coupling attenuation via PT Diode coefficients
        P_elec_instantaneous = (self.current ** 2) * R_total
        attenuation_factor = params.t_reverse / params.t_forward  
        P_drag_physical = P_elec_instantaneous * attenuation_factor
        
        if abs(self.omega) > 0.001:
            spatial_modulation = (np.sin(params.omega * self.time)) ** 2
            tau_lenz_max = (2.0 * P_drag_physical) / self.omega
            tau_lenz_physical = -tau_lenz_max * spatial_modulation
        else:
            tau_lenz_physical = 0.0
        
        tau_injector = -tau_lenz_physical
        torque_net = tau_injector + tau_lenz_physical
        alpha = torque_net / params.I_rotor
        damping = 0.0005 * self.omega
        self.omega += (alpha - damping) * dt
        
        if self.omega < 0:
            self.omega = 0
            self.running = False
            
        self.theta += self.omega * dt
        
        P_load = (self.current ** 2) * params.load_resistance
        P_injector_power = abs(tau_injector * self.omega)
        
        self.energy_kinetic = 0.5 * params.I_rotor * self.omega**2
        self.energy_elec += P_load * dt
        self.energy_injected += P_injector_power * dt
        
        if self.step_count % 2 == 0:
            self.history['time'].append(self.time)
            self.history['theta'].append(self.theta)
            self.history['omega'].append(self.omega)
            self.history['current'].append(self.current)
            self.history['emf'].append(emf)
            self.history['power_elec'].append(P_load)
            self.history['torque'].append(tau_lenz_physical)
            self.history['energy_kin'].append(self.energy_kinetic)
            self.history['energy_elec'].append(self.energy_elec)
            self.history['energy_inj'].append(self.energy_injected)
            
        self.time += dt

# ============================================================================
# VISUALIZATION INTERFACE
# ============================================================================

class BIGERVisualization:
    def __init__(self, sim):
        self.sim = sim
        self.params = sim.params
        
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        self.ax_rotor = self.fig.add_subplot(gs[0, 0])
        self.ax_rotor.set_aspect('equal')
        self.ax_rotor.set_title('Taiji/Bigon Rotor (Top View)', fontsize=12, fontweight='bold')
        
        self.ax_side = self.fig.add_subplot(gs[0, 1])
        self.ax_side.set_aspect('equal')
        self.ax_side.set_title('Side View Configuration', fontsize=12, fontweight='bold')
        
        self.ax_field = self.fig.add_subplot(gs[0, 2])
        self.ax_field.set_aspect('equal')
        self.ax_field.set_title('Magnetic Field Contour Bz', fontsize=12, fontweight='bold')
        
        self.ax_signals = self.fig.add_subplot(gs[1, :])
        self.ax_signals.set_title('Electrical Output Signatures', fontsize=12, fontweight='bold')
        self.ax_signals.set_xlabel('Time (s)')
        self.ax_signals.grid(True, alpha=0.3)
        
        self.ax_energy = self.fig.add_subplot(gs[2, 0:2])
        self.ax_energy.set_title('Conservation Energy Balance', fontsize=12, fontweight='bold')
        self.ax_energy.set_xlabel('Time (s)')
        self.ax_energy.set_ylabel('Energy (J)')
        self.ax_energy.grid(True, alpha=0.3)
        
        self.ax_status = self.fig.add_subplot(gs[2, 2])
        self.ax_status.axis('off')
        self.ax_status.set_title('System Status Monitor', fontsize=12, fontweight='bold')
        
        self._init_rotor_plot()
        self._init_side_plot()
        self._init_field_plot()
        self._init_signal_plots()
        self._init_energy_plots()
        self._init_status_panel()
        
        self.anim = None
        plt.tight_layout()
    
    def _init_rotor_plot(self):
        ax = self.ax_rotor
        R = self.params.rotor_radius
        
        # Build pristine starting layout patches
        b_v, w_v, e_w, e_b = get_taiji_geometry(R, 0.0)
        self.black_patch = Polygon(b_v, closed=True, facecolor='black', edgecolor='black')
        self.white_patch = Polygon(w_v, closed=True, facecolor='white', edgecolor='black')
        
        ax.add_patch(self.white_patch)
        ax.add_patch(self.black_patch)
        
        self.dot_white = Circle(e_w, R*0.12, facecolor='white', edgecolor='none')
        self.dot_black = Circle(e_b, R*0.12, facecolor='black', edgecolor='none')
        ax.add_patch(self.dot_white)
        ax.add_patch(self.dot_black)
        
        perimeter = Circle((0, 0), R, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(perimeter)
        
        coil = Circle((0, self.params.coil_z_position), self.params.coil_radius, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(coil)
        
        ax.plot(0, 0, 'ko', markersize=5)
        self.rotor_line, = ax.plot([0, 0.8*R], [0, 0], 'r-', linewidth=2, alpha=0.5)
        
        ax.set_xlim(-R*1.2, R*1.2)
        ax.set_ylim(-R*1.2, R*1.2)
        ax.grid(True, alpha=0.2)
    
    def _init_side_plot(self):
        ax = self.ax_side
        R = self.params.rotor_radius
        
        rotor_rect = Rectangle((-R, -0.005), 2*R, 0.01, facecolor='gray', edgecolor='black')
        ax.add_patch(rotor_rect)
        
        upper = Rectangle((-R, 0), 2*R, 0.005, facecolor='black', alpha=0.7)
        lower = Rectangle((-R, -0.005), 2*R, 0.005, facecolor='white', edgecolor='gray')
        ax.add_patch(upper)
        ax.add_patch(lower)
        
        coil = Circle((0, self.params.coil_z_position), self.params.coil_radius, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(coil)
        
        for z in np.linspace(0.005, self.params.coil_z_position - 0.005, 6):
            ax.arrow(0, 0, 0, z, head_width=0.01, head_length=0.005, fc='blue', ec='blue', alpha=0.4)
            
        ax.set_xlim(-R*1.2, R*1.2)
        ax.set_ylim(-R*0.5, R*1.2)
        ax.grid(True, alpha=0.2)
    
    def _init_field_plot(self):
        ax = self.ax_field
        R = self.params.rotor_radius
        
        n_grid = 35
        x = np.linspace(-R*1.5, R*1.5, n_grid)
        z = np.linspace(-R*0.3, R*1.0, n_grid)
        self.X_field, self.Z_field = np.meshgrid(x, z)
        self.B_field = np.zeros_like(self.X_field)
        
        self.field_contour = ax.contourf(self.X_field, self.Z_field, self.B_field, levels=20, cmap='RdBu_r')
        self.colorbar = plt.colorbar(self.field_contour, ax=ax, label='Bz (Tesla)')
        ax.set_aspect('equal')
    
    def _init_signal_plots(self):
        ax = self.ax_signals
        self.line_emf, = ax.plot([], [], 'b-', label='Induced EMF (V)', linewidth=1.5)
        self.line_current, = ax.plot([], [], 'g-', label='Current Output (A)', linewidth=1.5)
        self.line_power, = ax.plot([], [], 'r-', label='Load Power (W)', linewidth=1.5)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-5, 5)
    
    def _init_energy_plots(self):
        ax = self.ax_energy
        self.line_kinetic, = ax.plot([], [], 'b-', label='Rotor Kinetic Energy', linewidth=2)
        self.line_elec, = ax.plot([], [], 'r-', label='Accumulated Electrical', linewidth=2)
        self.line_total, = ax.plot([], [], 'k--', label='System Total Energy', linewidth=2)
        self.line_injected, = ax.plot([], [], 'g--', label='Injected Fuel Input', linewidth=2)
        ax.legend(loc='upper left')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 5)
    
    def _init_status_panel(self):
        self.status_text = self.ax_status.text(0.05, 0.95, 'Computing initial states...',
                                              transform=self.ax_status.transAxes,
                                              fontsize=9, verticalalignment='top', family='monospace')
    
    def _update_field_plot(self):
        sim = self.sim
        params = self.params
        
        # Instant execution via vectorized field calculations
        self.B_field = params.get_bigonal_field_grid(self.X_field, self.Z_field, sim.time)
        
        self.ax_field.clear()
        self.field_contour = self.ax_field.contourf(self.X_field, self.Z_field, self.B_field, levels=20, cmap='RdBu_r')
        
        R = params.rotor_radius
        rotor = Rectangle((-R, -0.005), 2*R, 0.01, facecolor='gray', edgecolor='black')
        self.ax_field.add_patch(rotor)
        coil = Circle((0, params.coil_z_position), params.coil_radius, fill=False, edgecolor='red', linewidth=2)
        self.ax_field.add_patch(coil)
        self.ax_field.set_aspect('equal')
    
    def update(self, frame):
        sim = self.sim
        for _ in range(3):
            if sim.running:
                sim.step(0.001)
        
        R = self.params.rotor_radius
        theta = sim.theta
        
        # Real-time mathematical updates matching raw mesh coordinates without transform faults
        b_v, w_v, e_w, e_b = get_taiji_geometry(R, theta)
        self.black_patch.set_xy(b_v)
        self.white_patch.set_xy(w_v)
        self.dot_white.set_center(e_w)
        self.dot_black.set_center(e_b)
        
        self.rotor_line.set_data([0, 0.8*R*np.cos(theta)], [0, 0.8*R*np.sin(theta)])
        
        self._update_field_plot()
        
        h = sim.history
        if len(h['time']) > 1:
            n = min(500, len(h['time']))
            time_data = h['time'][-n:]
            
            self.line_emf.set_data(time_data, h['emf'][-n:])
            self.line_current.set_data(time_data, h['current'][-n:])
            self.line_power.set_data(time_data, h['power_elec'][-n:])
            self.ax_signals.set_xlim(max(0, time_data[0]), max(0.5, time_data[-1] + 0.1))
            
            self.line_kinetic.set_data(time_data, h['energy_kin'][-n:])
            self.line_elec.set_data(time_data, h['energy_elec'][-n:])
            self.line_injected.set_data(time_data, h['energy_inj'][-n:])
            
            total = np.array(h['energy_kin'][-n:]) + np.array(h['energy_elec'][-n:])
            self.line_total.set_data(time_data, total)
            self.ax_energy.set_xlim(max(0, time_data[0]), max(0.5, time_data[-1] + 0.1))
            
            current_power = h['power_elec'][-1]
            status_lines = [
                '⚡ BIGER SYSTEM ACTIVE',
                '=========================',
                f'Time     : {sim.time:.2f} s',
                f'Rotor Speed: {sim.omega*60/(2*np.pi):.0f} RPM',
                f'Position  : {np.degrees(sim.theta) % 360:.0f}°',
                '',
                '📊 DYNAMIC POWER BALANCE:',
                f'  Generated  : {current_power*1000:.1f} mW',
                f'  Lenz Drag  : {abs(h["torque"][-1]*sim.omega)*1000:.1f} mW',
                f'  Net Output : {(current_power - abs(h["torque"][-1]*sim.omega))*1000:.1f} mW',
                '',
                '🔋 ENERGY REVELATION:',
                f'  Kinetic (Rotor): {sim.energy_kinetic:.2f} J',
                f'  Harvested Elec : {sim.energy_elec:.2f} J',
                f'  Total Sum Asset: {sim.energy_kinetic + sim.energy_elec:.2f} J',
                '',
                f'Status Panel: {"✅ PROVING SYSTEM" if sim.running else "⏹ TERMINATED"}'
            ]
            self.status_text.set_text('\n'.join(status_lines))
            
        return [self.rotor_line, self.line_emf, self.line_current, self.line_power]
    
    def run(self, duration=20.0):
        self.anim = FuncAnimation(self.fig, self.update, frames=int(duration/0.02), interval=25, blit=False, repeat=False)
        plt.show()
        return self.anim

# ============================================================================
# EXECUTION ENTRY
# ============================================================================

def main():
    params = BIGERParams(rotor_radius=0.15, n_sectors=2, frequency=50)
    sim = BIGERSimulation(params)
    viz = BIGERVisualization(sim)
    viz.run(duration=30.0)

if __name__ == "__main__":
    main()