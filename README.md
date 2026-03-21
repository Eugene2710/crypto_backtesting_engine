## Strategies

### SMA 20/50 Crossover
The idea is: don't predict where price is going, just follow the trend once it's established.

- SMA20 (fast) reacts quickly to recent price moves                                                                                                                                
- SMA50 (slow) reflects the longer-term trend
- When SMA20 crosses above SMA50 → recent price momentum has outpaced the longer trend → interpreted as the start of an uptrend → enter long                                       
- When SMA20 crosses below SMA50 → short-term momentum has fallen below the longer trend → interpreted as the start of a downtrend → exit long  

Limitation
1. Lagging by design
   
   SMAs are calculated on past closes. By the time a crossover fires, the move has already partly happened. 
   
   You are always late to the party.


2. Whipsaw in sideways markets
   
   In a ranging/choppy market(no trend), SMA20 ad SMA50 weave back and forth repeatedly, generating false signals and rack up fees with no directional payoff. 
   
   This is the BIGGEST STRUCTURAL WEAKNESS.


3. No volatility awareness
   
   The signal fires regardless of whether the market is volatile or dead calm.
   
   A crossover in a thin, low-volume period is treated identically to one in a high-conviction breakout.


4. Parameter sensitivity 
   
   20 and 50 are conventional but arbitrary. Backtesting will show the strategy looks great on the specific window chosen or not
   - a curve-fitting risk

5. Single timeframe
   
   Signal uses only daily closes currently - a trend that is obvious on a weekly chart or already reversed on a 4H chart is invisible to this strategy.


6. Long-only
   
   Cannot profit from downtrends - half of all market time is left on the table.



Strategy (signal logic)     ← market-agnostic, works anywhere                                                                                                                      
      ↓                                                                                                                                                                              
Portfolio (position sizing)  ← currently assumes cash account, no leverage
      ↓                                                                                                                                                                              
Broker (fill simulation)     ← currently assumes spot: pay cash, receive asset                                                                                                     
                                  no funding rate, no margin, no liquidation price  