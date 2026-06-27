from __future__ import annotations

import re

from .data import Question


ZH_EN_TERMS = {
    "黑洞": "black hole Schwarzschild radius event horizon",
    "事件视界": "event horizon Schwarzschild radius",
    "太阳质量": "solar mass",
    "X 射线": "X ray X-ray Bragg diffraction crystallography",
    "X射线": "X ray X-ray Bragg diffraction crystallography",
    "岩盐": "NaCl sodium chloride rock salt lattice spacing Bragg",
    "强极大值": "constructive interference maximum Bragg",
    "波长": "wavelength lambda",
    "不确定度": "uncertainty Heisenberg momentum position",
    "结合能": "binding energy mass defect nuclear binding energy",
    "原子质量": "atomic mass mass defect",
    "正向偏置": "forward bias diode semiconductor",
    "反向偏置": "reverse bias diode semiconductor",
    "半衰期": "half life decay rate",
    "光子": "photon flux quantum",
    "折射": "refraction Snell law refractive index",
    "入射角": "angle of incidence Snell law",
    "理想气体": "ideal gas kinetic energy Boltzmann",
    "平动动能": "translational kinetic energy",
    "熵": "entropy",
    "焓": "enthalpy",
    "吉布斯": "Gibbs free energy",
    "亥姆霍兹": "Helmholtz free energy",
    "化学势": "chemical potential Gibbs-Duhem equation",
    "摩尔分数": "mole fraction Gibbs-Duhem equation",
    "配分函数": "partition function",
    "转动配分函数": "rotational partition function asymmetric rotor",
    "不对称转子": "asymmetric rotor rotational partition function",
    "玻尔兹曼": "Boltzmann distribution",
    "薛定谔": "Schrodinger equation",
    "谐振子": "harmonic oscillator",
    "粒子在箱中": "particle in a box",
    "转动": "rotational energy rotor",
    "振动": "vibrational energy oscillator",
    "晶格": "lattice crystal",
    "衍射": "diffraction Bragg",
    "半径": "radius",
    "角": "angle theta",
    "温度": "temperature",
    "压力": "pressure",
    "体积": "volume",
    "速度": "speed velocity",
    "仰角": "projectile motion launch angle",
    "小车": "relative motion projectile",
    "无摩擦": "conservation of energy frictionless normal force",
    "定压热容": "heat capacity constant pressure enthalpy",
    "热容": "heat capacity enthalpy temperature dependence",
    "焓增加": "enthalpy change heat capacity integral",
    "电对": "standard electrode potential electrochemical series",
    "标准电势": "standard electrode potential",
    "吸收系数": "Beer Lambert law absorption coefficient",
    "光强": "Beer Lambert law intensity absorption",
    "德布罗意": "de Broglie wavelength kinetic energy",
    "角动量量子数": "orbital angular momentum quantum number",
    "偶极矩": "dipole moment",
    "正电子素": "positronium reduced mass ground state energy",
    "磁矩": "magnetic moment Bohr magneton",
    "自旋": "spin angular momentum magnetic moment",
    "离解能": "dissociation energy potential energy curve",
    "势能": "potential energy curve dissociation energy",
    "夏至": "solar radiation flux insolation solar declination zenith angle latitude",
    "太阳辐射": "solar radiation flux insolation irradiance projection zenith angle",
    "太阳能收集器": "solar collector radiation flux insolation projection",
    "纬度": "latitude declination zenith angle",
    "光分": "Lorentz transformation spacetime interval time interval relativity",
    "惯性系": "inertial reference frame Lorentz transformation relativity",
    "观察者": "observer reference frame Lorentz transformation relativity",
    "凸面镜": "convex mirror spherical mirror focal length magnification image distance",
    "放大倍数": "magnification mirror equation image distance focal length",
    "反质子": "antiproton threshold energy relativistic collision center of mass",
    "固定质子靶": "fixed target threshold energy center of mass relativistic collision",
    "平均分子速率": "mean molecular speed Maxwell Boltzmann distribution rms speed molar mass",
    "氡": "radon molar mass mean molecular speed Maxwell distribution",
    "氢气": "hydrogen molar mass mean molecular speed Maxwell distribution",
    "弯道": "banked curve static friction centripetal force circular motion",
    "静摩擦": "static friction coefficient centripetal force banked curve",
    "梯子": "static equilibrium torque ladder normal force friction",
    "壁架": "static equilibrium torque ladder normal force",
    "最小动能": "minimum kinetic energy uncertainty principle confinement particle in a box",
}

GENERIC_TERMS = {
    "physics",
    "chemistry",
    "matter",
    "quan",
    "chemmc",
    "kinetics",
    "theorem",
    "quantum",
    "atomic",
    "wave",
    "optics",
    "thermodynamics",
    "electromagnetism",
    "mathrm",
    "text",
    "frac",
    "times",
    "mol",
    "ev",
    "cm",
    "kg",
    "ms",
    "pm",
    "nm",
    "pa",
}

LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")


def _latin_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in LATIN_TOKEN_RE.findall(text):
        lower = token.lower()
        if lower in GENERIC_TERMS or len(lower) <= 1:
            continue
        terms.append(token)
    return terms


def expand_query(question: Question) -> str:
    text = " ".join(
        part
        for part in [
            question.subfield or "",
            question.theorem or "",
            question.question,
        ]
        if part
    )
    expansions = [english for chinese, english in ZH_EN_TERMS.items() if chinese in text]
    if any(marker in text for marker in ("D_e", "D_0", "nu_e", "omega_e", "离解能")):
        expansions.append("dissociation energy zero point vibrational energy molecular potential")
    latin_terms = _latin_terms(text)
    return " ".join(expansions + latin_terms).strip()
