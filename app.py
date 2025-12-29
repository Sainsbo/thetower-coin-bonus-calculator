# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:16:44 2025

@author: aimio
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class Base:
    
    """
    Base class for the coin bonus calculations. Considers the global bonuses
    only, so will always return bonus, regardless of the time supplied
    
    Base class bonus should be everything shown in run on the 'all coins bonus'
    along with the coins_per_kill bonus from labs
    """
    
    def __init__(self, bonus):
        self.bonus = bonus
        
    def bonus_at_time_step(self, time):
        return self.bonus
    
    
class SpotLight(Base):
    
    """
    Used for SL only
    """
    
    def __init__(self, full_bonus, quantity, angle):
        self.bonus = full_bonus * ((quantity * angle) / 360)
   
    
class TimeVarying(Base):
    
    """
    Used for GT, DW and GB
    """
    
    def __init__(self, bonus, duration, cooldown):
        self.bonus = bonus
        self.duration = duration
        self.cooldown = cooldown
        
    def bonus_at_time_step(self, time):
        
        if time % self.cooldown < self.duration:
            return self.bonus
        else:
            return 1

class BlackHole(Base):
    
    """
    Accounts for BH coverage (quantity and size)
    """
    
    def __init__(self, tower_range, coin_bonus, duration, cooldown, size, 
                 quantity):
        self.tower_range= tower_range
        self.coin_bonus = coin_bonus
        self.duration = duration
        self.cooldown = cooldown
        self.size = size
        self.quantity = quantity
        self.bonus = self.coverage_fraction(self.tower_range, self.size, 
                                       self.quantity) * self.coin_bonus
        
    @staticmethod
    def coverage_fraction(tower_range, bh_size, bh_quant, 
                          samples=2_000_000, seed=0):
        
        rng = np.random.default_rng(seed)

        # Sample uniformly inside the main circle
        theta = rng.uniform(0, 2*np.pi, samples)
        radius = tower_range * np.sqrt(rng.uniform(0, 1, samples))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Small circle centres
        angles = np.linspace(0, 2*np.pi, bh_quant, endpoint=False)
        cx = 0.9 * tower_range * np.cos(angles)
        cy = 0.9 * tower_range * np.sin(angles)

        covered = np.zeros(samples, dtype=bool)

        for i in range(bh_quant):
            dx = x - cx[i]
            dy = y - cy[i]
            covered |= (dx*dx + dy*dy) <= bh_size*bh_size

        return covered.mean()
        
    
    def bonus_at_time_step(self, time):
        
        if time % self.cooldown < self.duration:
            return self.bonus
        else:
            return 1
        
class GoldBot(Base):
    
    def __init__(self, tower_range, coin_bonus, duration, cooldown, size):
        self.tower_range = tower_range
        self.coin_bonus = coin_bonus
        self.duration= duration
        self.cooldown = cooldown
        self.size = size
        self.bonus = self.gold_bot_coverage(tower_range, size) * self.coin_bonus
        
    @staticmethod
    def gold_bot_coverage(m, n, samples=2_000_000, seed=0):
        
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2*np.pi, samples)
        radius = m * np.sqrt(rng.uniform(0, 1, samples))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    
        d = np.sqrt(x**2 + y**2)
    
        area_inside = np.zeros(samples)
        fully_inside = d + n <= m
        area_inside[fully_inside] = np.pi * n**2

        partial = ~fully_inside
        R = m
        r2 = n
        d_partial = d[partial]

        term1 = r2**2 * np.arccos((d_partial**2 + r2**2 - 
                                   R**2)/(2*d_partial*r2))
        term2 = R**2 * np.arccos((d_partial**2 + R**2 - r2**2)/(2*d_partial*R))
        term3 = 0.5 * np.sqrt((-d_partial + r2 + R)*(d_partial + r2 - R)*
                              (d_partial - r2 + R)*(d_partial + r2 + R))
        area_inside[partial] = term1 + term2 - term3

        return np.mean(area_inside) / (np.pi * R**2)

    def bonus_at_time_step(self, time):
            
        if time % self.cooldown < self.duration:
            return self.bonus
        else:
            return 1
        

class TotalCoin:
    
    def __init__(self,
                 coin_per_kill_bonus,
                 base_coin_bonus,
                 tower_range,
                 sl_coin_bonus,
                 sl_quantity,
                 sl_angle,
                 bh_coin_bonus,
                 bh_duration,
                 bh_cooldown,
                 bh_size,
                 bh_quantity,
                 gt_coin_bonus,
                 gt_duration,
                 gt_cooldown,
                 dw_coin_bonus,
                 dw_duration,
                 dw_cooldown,
                 gb_coin_bonus,
                 gb_duration,
                 gb_cooldown,
                 gb_size):
        
        self.coin_per_kill_bonus = coin_per_kill_bonus
        self.base_coin_bonus = base_coin_bonus
        self.tower_range = tower_range
        
        self.sl_coin_bonus = sl_coin_bonus
        self.sl_quantity = sl_quantity
        self.sl_angle = sl_angle
        
        self.bh_coin_bonus = bh_coin_bonus
        self.bh_duration = bh_duration
        self.bh_cooldown = bh_cooldown
        self.bh_size = bh_size
        self.bh_quantity = bh_quantity
        
        self.gt_coin_bonus = gt_coin_bonus
        self.gt_duration = gt_duration
        self.gt_cooldown = gt_cooldown
        
        self.dw_coin_bonus = dw_coin_bonus
        self.dw_duration = dw_duration
        self.dw_cooldown = dw_cooldown
        
        self.gb_coin_bonus = gb_coin_bonus
        self.gb_duration = gb_duration
        self.gb_cooldown = gb_cooldown
        self.gb_size = gb_size
        
    def mc_estimator(self, title):
            
        """
        Estimates the total coin bonus (time-averaged) taking into account
        UW and bot cooldown, duration and sync
        """
            
        BaseBonus = Base(bonus = self.coin_per_kill_bonus * 
                         self.base_coin_bonus)
        SpotLightBonus = SpotLight(self.sl_coin_bonus, self.sl_quantity, 
                                       self.sl_angle)
        BlackHoleBonus = BlackHole(self.tower_range, self.bh_coin_bonus, 
                                   self.bh_duration, self.bh_cooldown, 
                                   self.bh_size, self.bh_quantity)
        GoldenTowerBonus = TimeVarying(self.gt_coin_bonus, self.gt_duration, 
                                           self.gt_cooldown)
        DeathWaveBonus = TimeVarying(self.dw_coin_bonus, self.dw_duration, 
                                         self.dw_cooldown)
        GoldBotBonus = GoldBot(self.tower_range, self.gb_coin_bonus, 
                               self.gb_duration, self.gb_cooldown, 
                               self.gb_size)
            
        instant_bonus = []
        for i in np.arange(0,100000,0.5):
            base_bonus = BaseBonus.bonus_at_time_step(i)
            sl_bonus = SpotLightBonus.bonus_at_time_step(i)
            bh_bonus = BlackHoleBonus.bonus_at_time_step(i)
            gt_bonus = GoldenTowerBonus.bonus_at_time_step(i)
            dw_bonus = DeathWaveBonus.bonus_at_time_step(i)
            gb_bonus = GoldBotBonus.bonus_at_time_step(i)
            instant_bonus.append(
                    base_bonus *
                    sl_bonus *
                    bh_bonus * 
                    gt_bonus *
                    dw_bonus *
                    gb_bonus)
            
        fig, axs = plt.subplots(1, 2, figsize = [12,6])
        axs[0].plot(np.arange(0,300,0.5), instant_bonus[0:600])
        axs[0].set_xlabel('time (seconds)')
        axs[0].set_ylabel('Total coin bonus')
        axs[1].plot(np.arange(0,3000,0.5), instant_bonus[0:6000])
        axs[1].set_xlabel('time (seconds)')
        axs[1].set_ylabel('Total coin bonus')
        plt.suptitle(f'mean across simulation = {np.mean(instant_bonus):.1f} \n {title}')
                
        return fig, np.mean(instant_bonus)
    
st.title("Coin Bonus Calculator")

with st.sidebar:
    st.header("Inputs")

    values = dict(
        coin_per_kill_bonus = st.number_input("Coin per kill bonus", 0.0, 10.0, 2.14),
        base_coin_bonus = st.number_input("Base coin bonus", 0.0, 1000.0, 332.35),
        tower_range = st.number_input("Tower range", 0.0, 200.0, 76.45),
        sl_coin_bonus = st.number_input("SL coin bonus", 0.0, 10.0, 3.0),
        sl_quantity = st.number_input("SL quantity", 0, 10, 3),
        sl_angle = st.number_input("SL angle", 0.0, 360.0, 48.0),
        bh_coin_bonus = st.number_input("BH coin bonus", 0.0, 50.0, 11.0),
        bh_duration = st.number_input("BH duration", 0.0, 200.0, 37.0),
        bh_cooldown = st.number_input("BH cooldown", 0.0, 500.0, 160.0),
        bh_size = st.number_input("BH size", 0.0, 150.0, 52.0),
        bh_quantity = st.number_input("BH quantity", 0.0, 3, 3, step = 1),
        gt_coin_bonus = st.number_input("GT coin bonus", 0.0, 50.0, 24.8),
        gt_duration = st.number_input("GT duration", 0.0, 200.0, 45.0),
        gt_cooldown = st.number_input("GT cooldown", 0.0, 500.0, 160.0),
        dw_coin_bonus = st.number_input("DW coin bonus", 0.0, 10.0, 2.25),
        dw_duration = st.number_input("DW duration", 0.0, 200.0, 25.0),
        dw_cooldown = st.number_input("DW cooldown", 0.0, 500.0, 160.0),
        gb_coin_bonus = st.number_input("GB coin bonus", 0.0, 10.0, 3.6),
        gb_duration = st.number_input("GB duration", 0.0, 200.0, 24.0),
        gb_cooldown = st.number_input("GB cooldown", 0.0, 500.0, 80.0),
        gb_size = st.number_input("GB size", 0.0, 100.0, 52.0)
    )
    
    
if st.button("Run simulation"):
    obj = TotalCoin(**values)
    fig, mean_bonus = obj.mc_estimator("Simulation results")
    st.pyplot(fig)
    st.metric("Average coin bonus", f"{mean_bonus:.1f}")

