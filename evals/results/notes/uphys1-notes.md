# 3.4 Motion with Constant Acceleration

*   **Notation Simplifications (Initial time $t_0 = 0$):**
    *   Elapsed time: $\Delta t = t$
    *   Displacement: $\Delta x = x - x_0$
    *   Change in velocity: $\Delta v = v - v_0$
    *   Subscript '0' denotes initial values; no subscript denotes final values.
*   **Constant Acceleration Assumption:** When acceleration ($a$) is constant, the average acceleration equals the instantaneous acceleration.
*   **Displacement and Position from Velocity:**
    *   Average velocity ($\overline{v}$): $\overline{v} = \frac{v_0 + v}{2}$
    *   Position equation: $x = x_0 + \overline{v}t$
*   **Solving for Final Velocity from Acceleration and Time:**
    *   Acceleration definition: $a = \frac{\Delta v}{\Delta t}$
    *   Final velocity equation: $v = v_0 + at$

**Key Terms:**
*   **Constant Acceleration:** Acceleration that does not change over time.
*   **Single-body motion:** Motion of one object.
*   **Two-body pursuit problems:** Motion involving two objects.
*   **Average Velocity ($\overline{v}$):** The average of the initial and final velocities when acceleration is constant.

**Equations:**
*   $\Delta t = t$
*   $\Delta x = x - x_0$
*   $\Delta v = v - v_0$
*   $\overline{v} = \frac{v_0 + v}{2}$
*   $x = x_0 + \overline{v}t$
*   $a = \frac{v - v_0}{t}$
*   $v = v_0 + at$

```mermaid
mindmap
  n0["Motion with Constant Acceleration (Section 3.4)"]
    n1["Learning Objectives"]
      n2["Identify which equations of motion are to be used to solve for unknowns."]
      n3["Use appropriate equations of motion to solve a two-body pursuit problem."]
    n4["Notation Simplifications (Initial time = 0)"]
      n5["$Δt = t$"]
      n6["$Δx = x - x_0$"]
      n7["$Δv = v - v_0$"]
      n8["Subscript 0: Initial values (position, velocity)."]
      n9["No subscript: Final values (time, position, velocity)."]
    n10["Key Assumption"]
      n11["Acceleration ($a$) is constant."]
      n12["Average acceleration = Instantaneous acceleration."]
    n13["Displacement and Position from Velocity"]
      n14["Average Velocity ($Δv$): $\overline(v) = \frac(v_0 + v)(2)$ (when acceleration is constant)."]
      n15["Position Equation: $x = x_0 + \overline(v)t$"]
      n16["Concept: $\overline(v)$ is the simple average of initial and final velocities."]
    n17["Solving for Final Velocity from Acceleration and Time"]
      n18["Definition of Acceleration: $a = \frac(Δv)(Δt)$"]
      n19["Derived Equation: $v = v_0 + at$"]
      n20["Example 3.7: Airplane landing calculation."]
    n21["Types of Motion Studied"]
      n22["Single-body motion"]
      n23["Two-body pursuit problems"]
```

---

# Section 5.3 Newton's Second Law

*   **Newton's second law** mathematically describes the cause-and-effect relationship between **force** and changes in motion.
*   It is a **quantitative** law used to calculate outcomes involving a force.
*   A **change in motion** is equivalent to a **change in velocity**, which means there is **acceleration**.
*   A **net external force** causes **nonzero acceleration**.
*   An **external force** acts on an object or system and originates outside of it.
*   An **internal force** acts between elements of the system.
*   Only **external forces** affect the motion of a system.
*   The relationship between acceleration ($\vec{a}$) and **net external force** ($\vec{F}_{\text{net}}$) is proportional: $\vec{a} \propto \vec{F}_{\text{net}}$.
*   The relationship between acceleration ($a$) and **mass** ($m$) is inversely proportional: $a \propto \frac{1}{m}$.
*   Experiments confirm that acceleration is exactly inversely proportional to mass and directly proportional to net external force.

```mermaid
mindmap
  n0["Newton's Second Law"]
    n1["Learning Objectives"]
      n2["Distinguish between external and internal forces"]
      n3["Describe Newton's second law of motion"]
      n4["Explain the dependence of acceleration on net force and mass"]
    n5["Key Concepts"]
      n6["Change in Motion $\iff$ Change in Velocity $\iff$ Acceleration"]
      n7["External Force"]
      n8["Internal Force"]
      n9["System of Interest"]
      n10["Net External Force ($\\vec(F)_(net)$)"]
    n11["Force Types"]
      n12["External Force: Acts on an object/system originating outside of it."]
      n13["Internal Force: Acts between elements of the system."]
      n14["Note: Only external forces affect the motion of a system."]
    n15["Relationships (Proportionalities)"]
      n16["Acceleration and Net External Force"]
      n17["$\vec(a) \propto \vec(F)_(net)$ (Acceleration is directly proportional to net external force)"]
      n18["Acceleration and Mass"]
      n19["$\text(a) \propto \frac(1)(m)$ (Acceleration is inversely proportional to mass)"]
    n20["Application"]
      n21["Quantitative law used to calculate situations involving a force."]
      n22["Requires defining system boundaries to identify external forces."]
```

---

# 6.2 Friction

*   **Friction** is a force that opposes relative motion between systems in contact.
*   Friction is a common yet complex force.
*   Friction arises in part due to the **roughness** of the surfaces in contact.
*   Part of the friction is due to **adhesive forces** between the surface molecules of the two objects.

### Types of Friction

*   **Static friction:** Acts between two systems that are in contact and **stationary** relative to one another.
*   **Kinetic friction:** Acts between two systems that are in contact and **moving** relative to one another.
*   Static friction is usually greater than kinetic friction.
*   Once an object is moving, it is easier to keep it moving than it was to get it started, meaning the kinetic frictional force is less than the static frictional force.

### Magnitude of Friction

*   **Static friction** is a responsive force that increases to be equal and opposite to whatever force is exerted, up to its maximum limit.
*   The maximum magnitude of static friction is given by:
    $$f_{\mathrm{s}}(\mathrm{max})=\mu_{\mathrm{s}}\,N$$
    where $\mu_{\mathrm{s}}$ is the **coefficient of static friction** and $N$ is the magnitude of the **normal force**.
*   The general condition for static friction is:
    $$f_{\mathrm{s}}\leq\mu_{\mathrm{s}}N$$
*   The magnitude of kinetic friction is given by:
    $$f_{\mathrm{k}}=\mu_{\mathrm{k}}N$$
    where $\mu_{\mathrm{k}}$ is the **coefficient of kinetic friction**.

**Comparison of Static and Kinetic Friction**

| Type of Friction | Condition | Description/Behavior | Governing Equation (Magnitude) |
|---|---|---|---|
| Static Friction | Objects are stationary relative to one another | Responds to applied force; increases to be equal and opposite to the push up to a maximum limit. | f_s(max) = μ_s N |
| Kinetic Friction | Objects are moving relative to one another | Opposes motion; once motion starts, it is generally easier to keep it moving than to start it. | f_k = μ_k N |

---

# 10.6 Torque

*   **Torque** is the rotational counterpart to force, related to changing the rotational motion of an object about an axis.
*   Torque has both **magnitude** and **direction**.
*   In rotation in a plane, torque can be either **clockwise** or **counterclockwise** relative to the chosen pivot point.
*   The magnitude of torque depends on the **lever arm** and the angle the force vector makes with the lever arm.
*   The **SI unit** of torque is **newtons times meters** ($\text{N}\cdot\text{m}$).
*   The **lever arm** is the perpendicular distance from the pivot point to the line determined by the force vector.

**Formulas:**

*   Torque vector: $\vec{\mathbf{\tau}}=\vec{\mathbf{r}}\times\vec{\mathbf{F}}$
*   Magnitude of torque: $\left|{\vec{\tau}}\right|=\left|{\vec{\bf r}}\,\times\,{\vec{\bf F}}\right|=rF{\sin\theta}$
*   Magnitude of torque using lever arm: $|\vec{\tau}|=r_{\perp}F$
*   Net torque: $\vec{\tau}_{\mathrm{net}}=\sum_{i}\vec{\tau}_{i}$

**Key Terms:**

*   **Torque**: The turning or twisting effectiveness of a force.
*   **Lever arm**: The perpendicular distance from the pivot point to the line determined by the force vector.
*   **Right-hand rule**: Used to determine the sign (direction) of a torque.

```mermaid
mindmap
  n0["Torque"]
    n1["Definition & Concept"]
      n2["Rotational counterpart to force"]
      n3["Related to changing rotational motion about an axis"]
      n4["Intuitive examples:"]
      n5["Door rotation (hinges)"]
      n6["Car accelerator"]
      n7["Body movement"]
    n8["Characteristics"]
      n9["Has magnitude and direction"]
      n10["Direction:"]
      n11["Clockwise or counterclockwise relative to pivot point"]
      n12["Magnitude depends on:"]
      n13["Lever arm (distance from pivot)"]
      n14["Angle between force vector and lever arm"]
    n15["Mathematical Description"]
      n16["Vector form (around O):"]
      n17["$\vec(\tau) = \vec(r) \times \vec(F)$"]
      n18["Magnitude form:"]
      n19["$\left|\vec(\tau)\right| = rF\sin\theta$"]
      n20["Lever arm form:"]
      n21["$|−\vec(\tau)| = r_(\perp)F$"]
      n22["SI Unit: newtons times meters (N\cdot m)"]
      n23["Sign determination:"]
      n24["Right-hand rule (cross product direction)"]
    n25["Net Torque"]
      n26["Sum of individual torques about a common axis"]
      n27["$\vec(\tau)_(net) = \sum_(i)\vec(\tau)_(i)$"]
      n28["Requires assigning appropriate signs (positive/negative)"]
```

---

# 13.2 Gravitation Near Earth's Surface

*   The acceleration of a free-falling object near Earth's surface is approximately $g$.
*   The force causing this acceleration is the **weight** of the object, with a value of $mg$.
*   Substituting $mg$ for the magnitude of the gravitational force in Newton's law of universal gravitation yields the scalar equation:
    $$mg=G\,\frac{mM_{\mathrm{E}}}{r^{2}}$$
*   The mass $m$ of the object cancels, resulting in:
    $$g=G\frac{M_{\mathrm{E}}}{r^{2}}$$
*   For objects near Earth's surface, the distance $r$ can be taken as the **radius of Earth** ($R_{\mathrm{E}}$).
*   The gravitational acceleration $g$ can be determined by knowing the mass of the astronomical body ($M$) and the distance ($r$) from its center.
*   The mass of the Moon ($M_{\mathrm{M}}$) can be estimated by assuming it has the same average density as Earth, using the ratio of volumes:
    $$\frac{M_{\mathrm{M}}}{M_{\mathrm{E}}}=\frac{R_{\mathrm{M}}^{3}}{R_{\mathrm{E}}^{3}}$$
*   The gravitational acceleration $g$ at a distance $r$ above Earth's surface can be calculated using the formula:
    $$g=G\frac{M_{\mathrm{E}}}{r^{2}}$$
*   Astronaut weightlessness in space stations is due to being in **free fall**, not the absence of gravity.
*   The vector form of the gravitational acceleration is:
    $${\bf\vec{g}}=G{\frac{M}{r^{2}}}{\bf\hat{r}}$$

**Key Terms:**
*   **Weight**
*   **Free fall**
*   **Radius of Earth**
*   **Gravitational acceleration**
*   **Vector field**

**Equations:**
$$mg=G\,\frac{mM_{\mathrm{E}}}{r^{2}}$$
$$g=G\frac{M_{\mathrm{E}}}{r^{2}}$$
$${\bf\vec{g}}=G{\frac{M}{r^{2}}}{\bf\hat{r}}$$

```mermaid
flowchart TD
    n0["Start: Observe free-falling object near Earth's surface"]
    n1["Identify acceleration of free-falling object as g (approx. 9.8 m/s²)"]
    n2["Relate weight (mg) to gravitational force using Newton's Law of Universal Gravitation: mg = G(mM_E)/r²"]
    n3["Cancel mass (m) to find the gravitational acceleration equation: g = G(M_E)/r²"]
    n4["Determine distance r (for surface objects, r ≈ Earth's radius)"]
    n5["Calculate Earth's mass (M_E) using known g, G, and r"]
    n6["To find mass of another body (e.g., Moon), use density assumption and volume ratio: M_M/M_E = (R_M/R_E)³"]
    n7["Calculate g at a different location (e.g., 400 km above Earth's surface) using the modified distance r = R_E + altitude"]
    n8["End: Gravitational acceleration is a scalar function of distance from the center of mass"]
    n0 --> n1
    n1 --> n2
    n2 --> n3
    n3 --> n4
    n4 --> n5
    n5 --> n6
    n6 --> n7
    n7 --> n8
```

---

# 14.5 Fluid Dynamics

*   **Fluid Dynamics** is the study of **fluids in motion**, contrasting with **fluid statics** (study of fluids at rest).
*   **Ideal fluid** is a fluid with negligible **viscosity** (internal friction).
*   For an **incompressible fluid**, the **density** is constant throughout, requiring a very large force to change its volume.
*   **Velocity vectors** represent fluid motion by indicating the speed and direction of the fluid at any point.
*   A **streamline** represents the path of a small volume of fluid, and the velocity is always tangential to it.
*   **Laminar flow** is characterized by smooth, parallel streamlines (sometimes called steady flow).
*   **No slip boundary conditions** are a special case of laminar flow where friction between the pipe and fluid is high, causing velocity to be greatest in the center and decrease near the walls.
*   **Turbulent flow** is characterized by irregular streamlines, mixing, and swirling, often occurring when fluid speed reaches a critical speed.
*   **Flow rate ($Q$)**, or **volume flow rate**, is the volume of fluid passing a given location through an area over a period of time.
*   The definition of flow rate is: $$Q=\frac{dV}{dt}$$
*   For a cylinder of cross-sectional area $A$ moving a distance $x$ in time $t$, the flow rate is: $$Q=\frac{dV}{dt}=\frac{d}{dt}(Ax)=A\frac{dx}{dt}=Av$$
*   The precise relationship between flow rate ($Q$) and average speed ($v$) is: $$Q=Av$$
*   Flow rate is directly proportional to both the average speed of the fluid and the cross-sectional area of the conduit.
*   The **equation of continuity** states that for any **incompressible fluid** (constant density) with no sources or sinks, the flow rate must be the same at all points along the pipe: $$Q_{1}=Q_{2}$$
*   This leads to the relationship: $$A_{1}\upsilon_{1}=A_{2}\upsilon_{2}$$

```mermaid
mindmap
  n0["Fluid Dynamics (Study of fluids in motion)"]
    n1["Fluid Types"]
      n2["Ideal Fluid"]
      n3["Incompressible Fluid"]
    n4["Flow Characteristics"]
      n5["Velocity Vectors"]
      n6["Streamlines"]
    n7["Flow Types (Streamlines)"]
      n8["Laminar Flow"]
      n9["Turbulent Flow"]
    n10["Flow Rate (Q)"]
      n11["Definition"]
      n12["Calculation"]
      n13["Relationship to Velocity"]
      n14["Equation of Continuity"]
```
