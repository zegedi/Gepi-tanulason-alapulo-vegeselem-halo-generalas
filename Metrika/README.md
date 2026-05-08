# Metrika

Ez a mappa a végeselem metrikák kiszámításához használt fájlokat tartalmazza. A metrikák definíciója, valamint az ezekhez tartozó hiba- és figyelmeztetési tartományok a [Modeling and Meshing Guide](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/pdf/ANSYS_Mechanical_APDL_Modeling_and_Meshing_Guide.pdf) alapján lettek meghatározva. A kiszámításhoz pedig a [Verdict](https://visit-sphinx-github-user-manual.readthedocs.io/en/3.4rc/_downloads/2a827d78bd7c1133b326536610130d03/VerdictManual-revA.pdf) könyvtárban bevezetett értékek lettek felhasználva.

## Implementált metrikák

A `metric.py` fájl az alábbi definíciókat és számítási módokat tartalmazza:

* **Aszpekt arány** (Aspect Ratio): [Dokumentáció](https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/corp/v251/en/wb_msh/msh_aspect_quad.html)
* **Jacobi hányados** (Jacobian Ratio): [Dokumentáció](https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/corp/v251/en/wb_msh/msh_jacobian_ratio.html)
* **Párhuzamos eltérés** (Parallel Deviation): [Dokumentáció](https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/corp/v251/en/wb_msh/msh_parallel_dev.html)
* **Maximális sarokszög** (Maximum Corner Angle): [Dokumentáció](https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/corp/v251/en/wb_msh/msh_max_corner_angle.html)

## Használat

A `quality.py` fájl segítségével határozhatók meg egy adott háló minőségi mutatói. A program bemenetként egy `.msh` formátumban megadott végeselem-hálót vár.
