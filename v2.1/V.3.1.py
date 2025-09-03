# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:37:15 2025

@author: USER
"""

from stl import mesh
import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, scrolledtext
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ========================
# Clase maya (para cargar STL)
# ========================
class maya():
    def __init__(self):
        self.model = None
        self.original_vectors = None
        self.normalized_vectors = None

    def file_load(self, filename: str):
        if not filename:
            raise ValueError("STL file not found")
        self.model = mesh.Mesh.from_file(filename)
        self.original_vectors = self.model.vectors.copy()
        self.normalized_vectors = None

    def get_normalized_vertices(self):
        if self.original_vectors is None:
            raise ValueError("No STL file loaded")
        if self.normalized_vectors is not None:
            return self.normalized_vectors
        
        model_flat = self.model.vectors.reshape(-1,3)
        max_val = np.max(np.abs(model_flat))   # Escalamos
        normalized = model_flat / max_val
        self.normalized_vectors = normalized.reshape(self.model.vectors.shape)
        return self.normalized_vectors
    

# ---------------- Rotations & Transformations ---------------- #
def rot_x(phi):
    s, c = math.sin(phi), math.cos(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(theta):
    s, c = math.sin(theta), math.cos(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(psi):
    s, c = math.sin(psi), math.cos(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def R_body_to_Fe(phi, theta, psi):
    return rot_z(psi) @ rot_y(theta) @ rot_x(phi)

def body_to_wind(alpha, beta):
    return rot_y(-alpha) @ rot_z(-beta)

def compute_angles(u, v, w, phi, theta):
    V = math.sqrt(u**2 + v**2 + w**2)
    alpha = math.atan2(w, u)
    beta = math.asin(v / V)
    gamma = theta - alpha
    return V, alpha, beta, gamma


# ---------------- Utility: NED -> Plot coords ---------------- #
def ned_to_plot(vec):
    return np.array([vec[0], vec[1], vec[2]])  


# ---------------- Plotting Function ---------------- #
def plot_case(nombre, u, v, w, phi, theta, psi,
              show_fe_axes=False,
              show_body_axes=True,
              show_wind_axes=True,
              show_velocity=True,
              enforce_visual_rhr=True,
              stl_model=None,
              view=(30,45)):
    
    v_body = np.array([u, v, w])
    Rnb = R_body_to_Fe(phi, theta, psi)
    v_Fe = Rnb @ v_body
    V, alpha, beta, gamma = compute_angles(u, v, w, phi, theta)

    # Bases
    e_fe = np.eye(3)
    e_body = np.eye(3)
    body_axes_ned = Rnb @ e_body
    wind_axes_ned = Rnb @ body_to_wind(alpha, beta).T @ np.eye(3)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d', facecolor='lightblue')

    scale = max(1.0, V/30)
    scale_axes = scale * (3 if "Case B" in nombre else 1)

    # ----------- Ejes Fe (NED) -----------
    if show_fe_axes:
        fe_x_plot = ned_to_plot(e_fe[:,0] * scale_axes)
        fe_y_plot = ned_to_plot(e_fe[:,1] * scale_axes)
        fe_z_plot = ned_to_plot(e_fe[:,2] * scale_axes)

        if enforce_visual_rhr:
            cross_fe = np.cross(fe_x_plot, fe_y_plot)
            norm = np.linalg.norm(cross_fe)
            if norm > 1e-9:
                fe_z_plot = cross_fe / norm * np.linalg.norm(fe_z_plot)

        ax.quiver(0, 0, 0, fe_x_plot[0], fe_x_plot[1], fe_x_plot[2], linestyle="dashed",color="r", label="X_Fe (North)")
        ax.quiver(0, 0, 0, fe_y_plot[0], fe_y_plot[1], fe_y_plot[2], linestyle="dashed",color="g", label="Y_Fe (East)")
        ax.quiver(0, 0, 0, fe_z_plot[0], fe_z_plot[1], fe_z_plot[2], linestyle="dashed",color="b", label="Z_Fe (Down)")

    # ----------- Ejes Body -----------
    if show_body_axes:
        xb_plot = ned_to_plot(body_axes_ned[:,0] * scale_axes)
        yb_plot = ned_to_plot(body_axes_ned[:,1] * scale_axes)
        zb_plot = ned_to_plot(body_axes_ned[:,2] * scale_axes)

        if enforce_visual_rhr:
            cross_b = np.cross(xb_plot, yb_plot)
            norm = np.linalg.norm(cross_b)
            if norm > 1e-9:
                zb_plot = cross_b / norm * np.linalg.norm(zb_plot)

        ax.quiver(0, 0, 0, xb_plot[0], xb_plot[1], xb_plot[2],linestyle="dashed", color="orange", label="Xb")
        ax.quiver(0, 0, 0, yb_plot[0], yb_plot[1], yb_plot[2],linestyle="dashed", color="purple", label="Yb")
        ax.quiver(0, 0, 0, zb_plot[0], zb_plot[1], zb_plot[2], linestyle="dashed",color="slategray", label="Zb")

    # ----------- Vector Wind -----------
    if show_wind_axes:
        wind_vec_ned = wind_axes_ned[:,0] * scale_axes
        wind_vec_plot = ned_to_plot(wind_vec_ned)
        ax.quiver(0, 0, 0,
                  wind_vec_plot[0], wind_vec_plot[1], wind_vec_plot[2], color="fuchsia", linewidth=2, label="Wind Vector")

    # ----------- Velocidad -----------
    if show_velocity and V > 1e-9:
        v_plot = ned_to_plot((v_Fe / V) * scale_axes * 1.2)
        ax.quiver(0, 0, 0, v_plot[0], v_plot[1], v_plot[2], color="aqua", linewidth=2, label="Velocity")

    # ----------- Modelo STL -----------
    if stl_model is not None:
        verts = stl_model.get_normalized_vertices()
        centroid = np.mean(verts.reshape(-1, 3), axis=0)
        verts_centered = verts - centroid  
        verts_centered[:, :, 0] *= -1
        verts_centered[:, :, 2] *= -1 
        verts_scaled = verts_centered * scale_axes

        transformed_faces = []
        for tri in verts_scaled:
            tri_rot = (Rnb @ tri.T).T  # rotación cuerpo->Fe
            tri_plot = [ned_to_plot(p) for p in tri_rot]
            transformed_faces.append(tri_plot)

        mesh_collection = Poly3DCollection(transformed_faces, alpha=0.6, facecolor="white", edgecolor="k")
        ax.add_collection3d(mesh_collection)

    texto = (f"α={math.degrees(alpha):.1f}°\n"
             f"β={math.degrees(beta):.1f}°\n"
             f"γ={math.degrees(gamma):.1f}°\n"
             f"φ={math.degrees(phi):.1f}°\n"
             f"θ={math.degrees(theta):.1f}°\n"
             f"ψ={math.degrees(psi):.1f}°")
    ax.text2D(0.05, 0.8, texto, transform=ax.transAxes, fontsize=12,
              bbox=dict(facecolor="white", alpha=0.7))

    ax.set_xlim(-1.5*scale_axes, 1.5*scale_axes)
    ax.set_ylim(-1.5*scale_axes, 1.5*scale_axes)
    ax.set_zlim(-1.5*scale_axes, 1.5*scale_axes)
    ax.set_xlabel("X (Fe) North")
    ax.set_ylabel("Y (Fe) East")
    ax.set_zlabel("Z (Fe) Down (visual)")
    ax.set_title(nombre)
    ax.legend()
    ax.view_init(elev=view[0], azim=view[1])
    plt.show()


# ---------------- Predefined Flight Cases ---------------- #
casos = {
    "Case A - No wind, straight & level": 
        {"u": 100, "v": 0, "w": 0, "phi": 0, "theta": 0, "psi": 0},
    "Case B - Crosswind present": 
        {"u": 80, "v": 20, "w": 0, "phi": math.radians(5), "theta": 0, "psi": math.radians(15)},
    "Case C - descend with wind": 
        {"u": 60, "v": 10, "w": 20, "phi": math.radians(5), "theta": math.radians(15), "psi": math.radians(0)},
    "Custom Case": None
}

views = {
    "Isometric": (210, 225),
    "Front": (180, 180),
    "Right Side": (180, 270),
    "Left Side": (180, 90),
    "Bottom": (90, 0),
    "Top": (-90, 0)
}


# ---------------- GUI ---------------- #
stl_model = maya()
stl_model.file_load("aircraft.stl")

def run_case(case_name, output_box, entries, show_fe):
    if case_name == "Custom Case":
        try:
            u = float(entries["u"].get())
            v = float(entries["v"].get())
            w = float(entries["w"].get())
            phi = math.radians(float(entries["phi"].get()))
            theta = math.radians(float(entries["theta"].get()))
            psi = math.radians(float(entries["psi"].get()))
        except ValueError:
            output_box.insert(tk.END, "⚠️ Please enter valid numbers!\n")
            return
    else:
        params = casos[case_name]
        u, v, w = params["u"], params["v"], params["w"]
        phi, theta, psi = params["phi"], params["theta"], params["psi"]

    V, alpha, beta, gamma = compute_angles(u, v, w, phi, theta)

    # Mostrar resultados en la caja de texto
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"--- {case_name} ---\n")
    output_box.insert(tk.END, f"Velocities (u,v,w) = {u, v, w}m/s\n")
    output_box.insert(tk.END, f"Velocity = {V}m/S\n")
    output_box.insert(tk.END, f"Alpha = {math.degrees(alpha):.2f}°\n")
    output_box.insert(tk.END, f"Beta  = {math.degrees(beta):.2f}°\n")
    output_box.insert(tk.END, f"Gamma = {math.degrees(gamma):.2f}°\n")
    output_box.insert(tk.END, f"Euler angles (phi,theta,psi) = {math.degrees(phi):.1f}, {math.degrees(theta):.1f}, {math.degrees(psi):.1f}\n")

    selected_view = views[view_var.get()]
    # Graficar
    plot_case(case_name, u, v, w, phi, theta, psi,
              show_fe_axes=show_fe,
              enforce_visual_rhr=True,
              stl_model=stl_model,
              view=selected_view) 

def on_case_change(event, entries_frame, entries):
    if case_var.get() == "Custom Case":
        entries_frame.pack(pady=10)
    else:
        entries_frame.pack_forget()


root = tk.Tk()
root.title("Aircraft Simulation - Coordinate Frames")

case_label = tk.Label(root, text="Select Flight Case:")
case_label.pack(pady=5)

case_var = tk.StringVar()
case_dropdown = ttk.Combobox(root, textvariable=case_var, values=list(casos.keys()), state="readonly")
case_dropdown.current(0)
case_dropdown.pack(pady=5)

entries_frame = tk.Frame(root)
labels = ["u", "v", "w", "phi (deg)", "theta (deg)", "psi (deg)"]
entries = {}
for lbl in labels:
    row = tk.Frame(entries_frame)
    lab = tk.Label(row, width=12, text=lbl+":", anchor="w")
    ent = tk.Entry(row, width=10)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
    lab.pack(side=tk.LEFT)
    ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[lbl.split()[0]] = ent

case_dropdown.bind("<<ComboboxSelected>>", lambda e: on_case_change(e, entries_frame, entries))

view_label = tk.Label(root, text="Select View:")
view_label.pack(pady=5)

view_var = tk.StringVar()
view_dropdown = ttk.Combobox(root, textvariable=view_var, values=list(views.keys()), state="readonly")
view_dropdown.current(0)  # por defecto "Isometric"
view_dropdown.pack(pady=5)

enforce_var = tk.BooleanVar(value=True)

output_box = scrolledtext.ScrolledText(root, width=60, height=10)
output_box.pack(pady=10)

run_button = tk.Button(root, text="Run Simulation", command=lambda: run_case(case_var.get(), output_box, entries, enforce_var.get()))
run_button.pack(pady=10)

root.mainloop()

