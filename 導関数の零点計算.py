print("--- スクリプト開始テスト ---")
import mpmath
from cypari2 import Pari
from typing import List, Tuple, Dict
import time

print("--- インポート完了 ---")

class RamanujanLFunction:
    """
    ラマヌジャンL関数 L(s, τ), およびその解析的に導出された導関数 L'(s, τ)
    の値を計算するためのクラス。
     L' の計算に数値積分を使用 
    """

    def __init__(self, precision=100):
        mpmath.mp.dps = precision
        self._tau_cache = {}
        self.pari = Pari()
        # キャッシュ
        self._gamma_cache = {}
        self._psi0_cache = {}
        self._psi1_cache = {}
        self._gamma_l_cache = {}
        self._gammainc_cache = {}
        self._gammainc_prime_integral_cache = {}
        print(f"--- PARI/GPとmpmathの精度を{precision}桁に設定 ---")

    def tau(self, n: int) -> int:
        if n in self._tau_cache: return self._tau_cache[n]
        if n <= 0: return 0
        result = int(self.pari.ramanujantau(n))
        self._tau_cache[n] = result
        return result

    # --- ヘルパー関数群 ---
    def _gamma(self, s):
        s = mpmath.mpc(s)
        if s in self._gamma_cache: return self._gamma_cache[s]
        if s.imag == 0 and s.real <= 0 and s.real == int(s.real): s += 1e-30j
        val = mpmath.gamma(s)
        self._gamma_cache[s] = val
        return val

    def _psi0(self, s):
        s = mpmath.mpc(s)
        if s in self._psi0_cache: return self._psi0_cache[s]
        if s.imag == 0 and s.real <= 0 and s.real == int(s.real): s += 1e-30j
        val = mpmath.psi(0, s)
        self._psi0_cache[s] = val
        return val

    def _psi1(self, s):
        s = mpmath.mpc(s)
        if s in self._psi1_cache: return self._psi1_cache[s]
        if s.imag == 0 and s.real <= 0 and s.real == int(s.real): s += 1e-30j
        val = mpmath.psi(1, s)
        self._psi1_cache[s] = val
        return val

    def _gamma_prime(self, s):
        return self._gamma(s) * self._psi0(s)

    def _gammainc(self, s, z):
        key = (mpmath.mpc(s), mpmath.mpc(z))
        if key in self._gammainc_cache: return self._gammainc_cache[key]
        s_comp = mpmath.mpc(s); z_comp = mpmath.mpc(z)
        if s_comp.imag == 0 and s_comp.real > 0 and s_comp.real == int(s_comp.real):
            s_comp += 1e-30j
        val = mpmath.gammainc(s_comp, z_comp, regularized=False)
        self._gammainc_cache[key] = val
        return val

    def _gammainc_prime(self, s, z):
        key = (mpmath.mpc(s), mpmath.mpc(z))
        if key in self._gammainc_prime_integral_cache:
            return self._gammainc_prime_integral_cache[key]
        s_val = mpmath.mpc(s); z_val = mpmath.mpc(z)
        try:
            integrand = lambda t: mpmath.exp(-t) * mpmath.power(t, s_val - 1) * mpmath.log(t)
            val = mpmath.quad(integrand, [z_val, mpmath.inf], maxdegree=8)
            self._gammainc_prime_integral_cache[key] = val
            return val
        except Exception as e:
            print(f"    Error in _gammainc_prime integral: {e}")
            return mpmath.nan
    # --- ヘルパー関数群ここまで ---

    def _gamma_times_L(self, s: mpmath.mpc, num_terms: int = 50) -> mpmath.mpc:
        s = mpmath.mpc(s)
        cache_key = (s, num_terms)
        if cache_key in self._gamma_l_cache: return self._gamma_l_cache[cache_key]
        if s.imag == 0 and s.real > 0 and s.real == int(s.real): s += 1e-30j
        k = 12; pi = mpmath.pi; sum1 = mpmath.mpc(0); sum2 = mpmath.mpc(0)
        for n in range(1, num_terms + 1):
            tau_n = self.tau(n); arg_z = 2 * pi * n
            sum1 += tau_n * mpmath.power(n, -s) * self._gammainc(s, arg_z)
            sum2 += tau_n * mpmath.power(n, s - k) * self._gammainc(k - s, arg_z)
        chi_s = mpmath.power(2 * pi, 2*s - k)
        result = sum1 + chi_s * sum2
        self._gamma_l_cache[cache_key] = result
        return result

    def L(self, s: mpmath.mpc, num_terms: int = 50) -> mpmath.mpc:
        s = mpmath.mpc(s)
        gamma_s = self._gamma(s)
        if gamma_s == 0: return mpmath.mpc('inf')
        gamma_times_l_val = self._gamma_times_L(s, num_terms=num_terms)
        if mpmath.isnan(gamma_times_l_val): return mpmath.nan
        return gamma_times_l_val / gamma_s

    def L_prime(self, s: mpmath.mpc, num_terms: int = 50) -> mpmath.mpc:
        s = mpmath.mpc(s)
        k = 12; pi = mpmath.pi; log2pi = mpmath.log(2*pi)
        gamma_s = self._gamma(s)
        gamma_prime_s = self._gamma_prime(s)
        if gamma_s == 0: return mpmath.mpc('inf')
        gamma_s_sq = gamma_s**2
        if gamma_s_sq == 0: return mpmath.mpc('inf')

        term1_sum = mpmath.mpc(0); term2_sum = mpmath.mpc(0); term3_sum = mpmath.mpc(0)
        term4_sum = mpmath.mpc(0); term5_sum = mpmath.mpc(0); term6_sum = mpmath.mpc(0)

        for n in range(1, num_terms + 1):
            tau_n = self.tau(n); n_pow_s = mpmath.power(n, -s)
            n_pow_12ms = mpmath.power(n, s - k)
            log_n = mpmath.log(n) if n > 1 else 0
            arg_z = 2 * pi * n
            gam_s_z = self._gammainc(s, arg_z)
            gam_prime_s_z = self._gammainc_prime(s, arg_z)
            gam_ks_z = self._gammainc(k - s, arg_z)
            gam_prime_ks_z = self._gammainc_prime(k - s, arg_z)
            if mpmath.isnan(gam_prime_s_z) or mpmath.isnan(gam_prime_ks_z): return mpmath.nan
            gam_prime_12ms_z = -gam_prime_ks_z

            term1_sum += tau_n * n_pow_s * gam_s_z
            term2_sum += tau_n * (-log_n) * n_pow_s * gam_s_z
            term3_sum += tau_n * n_pow_s * gam_prime_s_z
            term4_sum += tau_n * n_pow_12ms * gam_ks_z
            term5_sum += tau_n * log_n * n_pow_12ms * gam_ks_z
            term6_sum += tau_n * n_pow_12ms * gam_prime_12ms_z

        coeff_term1 = -gamma_prime_s / gamma_s_sq
        coeff_term2 = 1 / gamma_s
        coeff_term3 = 1 / gamma_s
        factor_2pi = mpmath.power(2 * pi, 2*s - k)
        coeff_term4_part1 = (2 * log2pi * factor_2pi * gamma_s - factor_2pi * gamma_prime_s) / gamma_s_sq
        coeff_term5 = factor_2pi / gamma_s
        coeff_term6 = factor_2pi / gamma_s

        result = (coeff_term1 * term1_sum + coeff_term2 * term2_sum + coeff_term3 * term3_sum +
                  coeff_term4_part1 * term4_sum + coeff_term5 * term5_sum + coeff_term6 * term6_sum)
        return result

    def L_double_prime(self, s: mpmath.mpc, num_terms: int = 50) -> mpmath.mpc:
        s = mpmath.mpc(s)
        k = 12; pi = mpmath.pi; log2pi = mpmath.log(2*pi)
        gamma_s = self._gamma(s)
        if gamma_s == 0: return mpmath.mpc('inf')
        gamma_prime_s = self._gamma_prime(s)
        gamma_double_prime_s = self._gamma_double_prime(s) # Helper needed
        if mpmath.isnan(gamma_double_prime_s): return mpmath.nan
        gamma_s_sq = gamma_s**2
        gamma_s_cub = gamma_s**3
        if gamma_s_sq == 0 or gamma_s_cub == 0: return mpmath.mpc('inf')

        sum_G = mpmath.mpc(0); sum_logG = mpmath.mpc(0); sum_Gp = mpmath.mpc(0)
        sum_log2G = mpmath.mpc(0); sum_logGp = mpmath.mpc(0); sum_Gpp = mpmath.mpc(0)
        sum_G12 = mpmath.mpc(0); sum_logG12 = mpmath.mpc(0); sum_Gp12 = mpmath.mpc(0)
        sum_log2G12 = mpmath.mpc(0); sum_logGp12 = mpmath.mpc(0); sum_Gpp12 = mpmath.mpc(0)

        for n in range(1, num_terms + 1):
            tau_n = self.tau(n); n_pow_s = mpmath.power(n, -s)
            n_pow_12ms = mpmath.power(n, s - k)
            log_n = mpmath.log(n) if n > 1 else 0
            log_n_sq = log_n**2 if n > 1 else 0
            arg_z = 2 * pi * n

            gam_s_z = self._gammainc(s, arg_z)
            gam_prime_s_z = self._gammainc_prime(s, arg_z)
            gam_double_prime_s_z = self._gammainc_double_prime(s, arg_z) # Helper needed

            gam_ks_z = self._gammainc(k - s, arg_z)
            gam_prime_ks_z = self._gammainc_prime(k - s, arg_z)
            gam_double_prime_ks_z = self._gammainc_double_prime(k - s, arg_z) # Helper needed

            if any(mpmath.isnan(v) for v in [gam_prime_s_z, gam_double_prime_s_z, gam_prime_ks_z, gam_double_prime_ks_z]):
                return mpmath.nan

            gam_prime_12ms_z = -gam_prime_ks_z
            gam_double_prime_12ms_z = gam_double_prime_ks_z

            sum_G += tau_n * n_pow_s * gam_s_z
            sum_logG += tau_n * (-log_n) * n_pow_s * gam_s_z
            sum_Gp += tau_n * n_pow_s * gam_prime_s_z
            sum_log2G += tau_n * log_n_sq * n_pow_s * gam_s_z
            sum_logGp += tau_n * (-log_n) * n_pow_s * gam_prime_s_z
            sum_Gpp += tau_n * n_pow_s * gam_double_prime_s_z

            sum_G12 += tau_n * n_pow_12ms * gam_ks_z
            sum_logG12 += tau_n * log_n * n_pow_12ms * gam_ks_z
            sum_Gp12 += tau_n * n_pow_12ms * gam_prime_12ms_z
            sum_log2G12 += tau_n * log_n_sq * n_pow_12ms * gam_ks_z
            sum_logGp12 += tau_n * log_n * n_pow_12ms * gam_prime_12ms_z
            sum_Gpp12 += tau_n * n_pow_12ms * gam_double_prime_12ms_z

        A_prime = -gamma_prime_s / gamma_s_sq
        A_double = (2 * gamma_prime_s**2 - gamma_s * gamma_double_prime_s) / gamma_s_cub
        factor_2pi = mpmath.power(2 * pi, 2*s - k)
        d_factor_2pi_ds = factor_2pi * (2 * log2pi)
        C_factor = factor_2pi / gamma_s
        C_prime = (d_factor_2pi_ds * gamma_s - factor_2pi * gamma_prime_s) / gamma_s_sq
        
        num_C_prime = d_factor_2pi_ds * gamma_s - factor_2pi * gamma_prime_s
        den_C_prime = gamma_s_sq
        d_num_ds = ( (factor_2pi * (2*log2pi)**2 * gamma_s + d_factor_2pi_ds * gamma_prime_s) - \
                     (d_factor_2pi_ds * gamma_prime_s + factor_2pi * gamma_double_prime_s) )
        d_den_ds = 2 * gamma_s * gamma_prime_s
        C_double = (d_num_ds * den_C_prime - num_C_prime * d_den_ds) / den_C_prime**2

        term1 = A_double * sum_G
        term2 = A_prime * sum_logG
        term3 = A_prime * sum_Gp
        term4 = A_prime * sum_logG
        term5 = (1/gamma_s) * sum_log2G
        term6 = (1/gamma_s) * sum_logGp
        term7 = A_prime * sum_Gp
        term8 = (1/gamma_s) * sum_logGp
        term9 = (1/gamma_s) * sum_Gpp

        term10 = C_double * sum_G12
        term11 = C_prime * sum_logG12
        term12 = -C_prime * sum_Gp12
        term13 = C_prime * sum_logG12
        term14 = C_factor * sum_log2G12
        term15 = -C_factor * sum_logGp12
        term16 = -C_prime * sum_Gp12
        term17 = -C_factor * sum_logGp12
        term18 = C_factor * sum_Gpp12

        result = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 +
                  term10 + term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18)
        return result

    def _gamma_double_prime(self, s): # Helper for L''
        psi0_s = self._psi0(s)
        psi1_s = self._psi1(s)
        gamma_s = self._gamma(s)
        return gamma_s * (psi1_s + psi0_s**2)

    def _gammainc_double_prime(self, s, z): # Helper for L''
        key = (mpmath.mpc(s), mpmath.mpc(z))
        s_val = mpmath.mpc(s); z_val = mpmath.mpc(z)
        try:
            integrand = lambda t: mpmath.exp(-t) * mpmath.power(t, s_val - 1) * mpmath.log(t)**2
            val = mpmath.quad(integrand, [z_val, mpmath.inf], maxdegree=8)
            return val
        except Exception:
            return mpmath.nan

class ZeroFinder:
    """
    L'(s, τ)の零点を探索するクラス。
     指定された全領域を一発でチェックし、零点がある場合のみ分割探索する 
    """
    def __init__(self, l_function: RamanujanLFunction):
        self.l_func = l_function
        self.num_terms_l = 50

    def count_zeros_in_rect(self, s_min, s_max, t_min, t_max) -> int:
        """指定された長方形内の零点数を偏角の原理で計算"""
        path = [mpmath.mpc(s_min, t_min), mpmath.mpc(s_max, t_min),
                mpmath.mpc(s_max, t_max), mpmath.mpc(s_min, t_max), mpmath.mpc(s_min, t_min)]
        try:
            # L''/L' を積分
            integrand = lambda s: self.l_func.L_double_prime(s, num_terms=self.num_terms_l) / self.l_func.L_prime(s, num_terms=self.num_terms_l)
            # 領域が大きい場合、積分精度が必要
            integral_val = mpmath.quad(integrand, path, maxdegree=8)
            num_zeros = mpmath.nint(mpmath.re(integral_val / (2j * mpmath.pi)))
            return int(num_zeros)
        except Exception as e:
            print(f"      [Warn] Integration failed in large rect t=[{t_min:.1f}, {t_max:.1f}]: {e}")
            return -1 # エラー

    def find_candidates_one_shot(self,
                               sigma_range: Tuple[float, float],
                               t_range: Tuple[float, float],
                               fine_step: float = 1.0   # 詳細探索のステップ
                               ) -> List[Dict]:
        s_min, s_max = sigma_range
        t_min, t_max = t_range
        
        print(f"フェーズ2: 全領域 [σ={s_min}-{s_max}, t={t_min}-{t_max}] の一括積分チェック...")
        
        # 1. 全体で積分 (ここが時間かかるかもしれないが、空なら即終了)
        total_zeros = self.count_zeros_in_rect(s_min, s_max, t_min, t_max)
        print(f"  -> 領域全体の零点数 (偏角の原理): {total_zeros}")

        if total_zeros <= 0:
            print("  -> 零点なし (または計算不可)。詳細探索をスキップして終了します。")
            return []
        
        # 2. 零点がある場合のみ詳細スキャン (赤・青 長方形)
        print(f"  -> 零点が存在するため、詳細探索 (step={fine_step}) を開始します...")
        return self._scan_fine_mesh_overlapping(sigma_range, t_range, fine_step)

    def _scan_fine_mesh_overlapping(self, sigma_range, t_range, step) -> List[Dict]:
        """指定された範囲内を赤・青の長方形で詳細スキャン"""
        found_in_block = []
        s_min, s_max = sigma_range
        t_min, t_max = t_range
        width = s_max - s_min
        t_min_mp, t_max_mp, step_mp = mpmath.mpf(t_min), mpmath.mpf(t_max), mpmath.mpf(step)

        # 1. 赤い長方形
        ts_red = mpmath.arange(t_min_mp, t_max_mp, step_mp)
        for t in ts_red:
            s0, s1, s2, s3 = mpmath.mpc(s_min, t), mpmath.mpc(s_max, t), mpmath.mpc(s_max, t + step_mp), mpmath.mpc(s_min, t + step_mp)
            path = [s0, s1, s2, s3, s0]
            try:
                integrand = lambda s: self.l_func.L_double_prime(s, num_terms=self.num_terms_l) / self.l_func.L_prime(s, num_terms=self.num_terms_l)
                integral_val = mpmath.quad(integrand, path, maxdegree=7)
                num = mpmath.nint(mpmath.re(integral_val / (2j * mpmath.pi)))
                if num != 0:
                    print(f"      -> (Red) t~{float(t):.2f} で候補発見 ({int(num)}個)")
                    found_in_block.append({'x': s_min, 'y': t, 'w': width, 'h': step_mp, 'zeros': num, 'color': 'red'})
            except: pass

        # 2. 青い長方形 (t_min - step/2 から)
        start_blue = t_min_mp - step_mp / 2.0
        # t_max を確実にカバーするために少し余分に回す
        end_blue = t_max_mp + step_mp / 2.0 
        ts_blue = mpmath.arange(start_blue, end_blue, step_mp)
        for t in ts_blue:
            s0, s1, s2, s3 = mpmath.mpc(s_min, t), mpmath.mpc(s_max, t), mpmath.mpc(s_max, t + step_mp), mpmath.mpc(s_min, t + step_mp)
            path = [s0, s1, s2, s3, s0]
            try:
                integrand = lambda s: self.l_func.L_double_prime(s, num_terms=self.num_terms_l) / self.l_func.L_prime(s, num_terms=self.num_terms_l)
                integral_val = mpmath.quad(integrand, path, maxdegree=7)
                num = mpmath.nint(mpmath.re(integral_val / (2j * mpmath.pi)))
                if num != 0:
                    print(f"      -> (Blue) t~{float(t):.2f} で候補発見 ({int(num)}個)")
                    found_in_block.append({'x': s_min, 'y': t, 'w': width, 'h': step_mp, 'zeros': num, 'color': 'blue'})
            except: pass
            
        return found_in_block

    def find_zero_in_square(self, candidate_info, max_iter=100, tolerance=None):
        sigma = candidate_info['x']; t = candidate_info['y']
        width = candidate_info['w']; height = candidate_info['h']
        s_current = mpmath.mpc(sigma + width / 2.0, t + height / 2.0)
        if tolerance is None: tolerance = mpmath.power(10, -(mpmath.mp.dps - 3))
        
        print(f"  - 精密探索開始: 長方形中心 {s_current}")

        for i in range(max_iter):
            try:
                f_val = self.l_func.L_prime(s_current, num_terms=self.num_terms_l)
                f_prime_val = self.l_func.L_double_prime(s_current, num_terms=self.num_terms_l)
                if mpmath.isnan(f_val) or mpmath.isnan(f_prime_val): return None
                if abs(f_val) < tolerance:
                    return s_current
                if f_prime_val == 0: return None
                s_next = s_current - f_val / f_prime_val
                if abs(s_next - s_current) < tolerance:
                    return s_next
                s_current = s_next
            except: return None
        
        if abs(f_val) < tolerance * 100: return s_current
        return None

def main():
    PRECISION = 30
    SIGMA_MIN = 4.5
    SIGMA_MAX = 7.0
    T_MIN = -1.0
    T_MAX = 5.0
    STEP_SIZE = 0.125
    NUM_TERMS_L = 30

    print("ラマヌジャン標準L関数の導関数 L'(s) の零点探索 (一括チェック版)")
    print("-" * 60)
    print(f"探索領域: σ=[{SIGMA_MIN}, {SIGMA_MAX}], t=[{T_MIN}, {T_MAX}]")

    start_total_time = time.time()

    try:
        l_func = RamanujanLFunction(precision=PRECISION)
        finder = ZeroFinder(l_func)
        finder.num_terms_l = NUM_TERMS_L
    except Exception as e: print(f"Init Error: {e}"); return

    #  一括探索を実行 
    candidate_rects = finder.find_candidates_one_shot(
        (SIGMA_MIN, SIGMA_MAX), (T_MIN, T_MAX), STEP_SIZE
    )
    print("-" * 60)

    if not candidate_rects:
        print("零点は見つかりませんでした (一括チェックで 0 個判定)。")
    else:
        print(f"フェーズ3: 精密探索 ({len(candidate_rects)} 候補)...")
        found_zeros = []
        tol = mpmath.power(10, -(mpmath.mp.dps - 3))

        for info in candidate_rects:
            z = finder.find_zero_in_square(info, tolerance=tol)
            if z:
                # 範囲内チェック (一応)
                if SIGMA_MIN <= z.real <= SIGMA_MAX and T_MIN <= z.imag <= T_MAX:
                    found_zeros.append(z)
                    if abs(z.imag) > 1e-9:
                        conj_z = mpmath.conj(z)
                        if SIGMA_MIN <= conj_z.real <= SIGMA_MAX and T_MIN <= conj_z.imag <= T_MAX:
                             is_dup = False
                             for ez in found_zeros:
                                 if mpmath.almosteq(conj_z, ez, abs_eps=tol*10): is_dup = True; break
                             if not is_dup: found_zeros.append(conj_z)

        print("-" * 60)
        found_zeros.sort(key=lambda z: z.imag)
        unique_zeros = []
        if found_zeros:
            unique_zeros.append(found_zeros[0])
            for i in range(1, len(found_zeros)):
                is_dup = False
                for uz in unique_zeros:
                    if mpmath.almosteq(found_zeros[i], uz, abs_eps=tol*100): is_dup = True; break
                if not is_dup: unique_zeros.append(found_zeros[i])
        
        t_plus_zeros = [z for z in unique_zeros if z.imag >= -1e-9]
        print(f"発見された L' 零点 (t>=0): {len(t_plus_zeros)} 個")
        for z in t_plus_zeros:
            print(f"σ = {float(z.real):.8f},  t = {float(z.imag):.8f}")

    print(f"総計算時間: {time.time() - start_total_time:.2f} 秒")

if __name__ == "__main__":
    main()
