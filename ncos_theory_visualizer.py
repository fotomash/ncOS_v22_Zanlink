120, 95), (130, 90), (140, 105)],
                'E': [(140, 105), (160, 130), (180, 150)]
            }

            labels = {
                'PS': (5, 95, 'PS'),
                'SC': (10, 70, 'SC'),
                'AR': (20, 75, 'AR'),
                'ST': (50, 75, 'ST'),
                'Spring': (100, 65, 'Spring'),
                'Test': (110, 85, 'Test'),
                'LPS': (120, 95, 'LPS'),
                'SOS': (140, 105, 'SOS')
            }

            title = "Wyckoff Accumulation Schematic"

        else:
            # Distribution schematic
            phases = {
                'A': [(0, 100), (10, 130), (20, 125), (30, 115)],
                'B': [(30, 115), (50, 125), (70, 120), (90, 115)],
                'C': [(90, 115), (100, 135), (110, 115)],
                'D': [(110, 115), (120, 105), (130, 110), (140, 95)],
                'E': [(140, 95), (160, 70), (180, 50)]
            }

            labels = {
                'PSY': (5, 105, 'PSY'),
                'BC': (10, 130, 'BC'),
                'AR': (20, 125, 'AR'),
                'ST': (50, 125, 'ST'),
                'UTAD': (100, 135, 'UTAD'),
                'Test': (110, 115, 'Test'),
                'LPSY': (120, 105, 'LPSY'),
                'SOW': (140, 95, 'SOW')
            }

            title = "Wyckoff Distribution Schematic"

        # Plot phases
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']

        for i, (phase, points) in enumerate(phases.items()):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            ax.plot(x, y, linewidth=3, color=colors[i], label=f'Phase {phase}')

            # Fill phase areas
            if i < len(phases) - 1:
                ax.axvspan(x[0], x[-1], alpha=0.1, color=colors[i])

        # Add labels
        for key, (x, y, text) in labels.items():
            ax.annotate(text, xy=(x, y), xytext=(x, y+5),
                       fontsize=12, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7))
            ax.plot(x, y, 'o', markersize=10, color='yellow', 
                   markeredgecolor='black', markeredgewidth=2)

        # Add phase labels
        phase_positions = {'A': 15, 'B': 60, 'C': 100, 'D': 125, 'E': 150}
        for phase, x_pos in phase_positions.items():
            ax.text(x_pos, ax.get_ylim()[1]*0.95, f'Phase {phase}',
                   fontsize=14, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=self.bg_color, alpha=0.8))

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f'wyckoff_{phase_type}_schematic.png', dpi=300, bbox_inches='tight')
        plt.close()

        return f'wyckoff_{phase_type}_schematic.png'

    def plot_smc_structures(self, price_data, structures):
        """Plot SMC structures on price chart"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot candlesticks
        for idx, row in price_data.iterrows():
            color = '#26a69a' if row['close'] > row['open'] else '#ef5350'
            ax.plot([idx, idx], [row['low'], row['high']], 
                   color=color, linewidth=1, alpha=0.7)
            ax.plot([idx, idx], [row['open'], row['close']], 
                   color=color, linewidth=3)

        # Plot order blocks
        for ob in structures.get('order_blocks', []):
            rect = plt.Rectangle((ob['start_idx'], ob['low']), 
                               ob['end_idx'] - ob['start_idx'], 
                               ob['high'] - ob['low'],
                               facecolor='blue' if ob['type'] == 'bullish' else 'red',
                               alpha=0.3, edgecolor='none')
            ax.add_patch(rect)

        # Plot FVGs
        for fvg in structures.get('fvgs', []):
            rect = plt.Rectangle((fvg['idx']-0.5, fvg['low']), 
                               1, fvg['high'] - fvg['low'],
                               facecolor='green' if fvg['type'] == 'bullish' else 'orange',
                               alpha=0.3, edgecolor='none')
            ax.add_patch(rect)

        # Plot liquidity levels
        for liq in structures.get('liquidity', []):
            ax.axhline(y=liq['level'], color='yellow', 
                      linestyle='--', linewidth=2, alpha=0.7)
            ax.text(len(price_data)-1, liq['level'], 
                   liq['type'], fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow', alpha=0.7))

        ax.set_title('Smart Money Concepts Analysis', fontsize=18, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('smc_structures.png', dpi=300, bbox_inches='tight')
        plt.close()

        return 'smc_structures.png'

    def plot_confluence_heatmap(self, confluences, price_range):
        """Create a heatmap of confluence zones"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create price levels
        price_levels = np.linspace(price_range[0], price_range[1], 50)

        # Calculate confluence strength at each level
        heatmap_data = []

        for price in price_levels:
            row = []
            for theory in ['Wyckoff', 'SMC', 'MAZ', 'Hidden Orders']:
                strength = 0
                for conf in confluences:
                    if theory in conf['theories']:
                        distance = abs(conf['price'] - price) / price
                        if distance < 0.001:  # Within 0.1%
                            strength = max(strength, conf['strength'])
                row.append(strength)
            heatmap_data.append(row)

        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='hot',
                      extent=[0, 4, price_range[0], price_range[1]])

        # Labels
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(['Wyckoff', 'SMC', 'MAZ', 'Hidden Orders'])
        ax.set_ylabel('Price Level', fontsize=14)
        ax.set_title('Confluence Strength Heatmap', fontsize=18, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confluence Strength', fontsize=12)

        plt.tight_layout()
        plt.savefig('confluence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        return 'confluence_heatmap.png'

    def plot_trade_setup(self, setup_data):
        """Visualize a complete trade setup"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Price chart with setup
        price_data = setup_data['price_data']

        # Plot price
        ax1.plot(price_data.index, price_data['close'], 
                color='white', linewidth=2, label='Price')

        # Mark entry, SL, and TPs
        entry_idx = setup_data['entry_idx']
        ax1.plot(entry_idx, setup_data['entry'], 'o', 
                markersize=15, color='lime', label='Entry')

        # Stop loss
        ax1.axhline(y=setup_data['stop_loss'], color='red', 
                   linestyle='--', linewidth=2, label='Stop Loss')

        # Take profits
        colors = ['yellow', 'orange', 'green']
        for i, tp in enumerate(setup_data['take_profits']):
            ax1.axhline(y=tp, color=colors[i], linestyle='--', 
                       linewidth=1.5, label=f'TP{i+1}')

        # Mark confluence zones
        for conf in setup_data['confluences']:
            ax1.axhspan(conf['low'], conf['high'], 
                       alpha=0.2, color='cyan')

        # Annotations
        ax1.text(entry_idx, setup_data['entry'], 
                f"Entry: {setup_data['entry']:.2f}\nRR: {setup_data['risk_reward']:.1f}",
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
                                     facecolor='lime', alpha=0.8))

        ax1.set_title(f"{setup_data['pair']} - {setup_data['direction'].upper()} Setup",
                     fontsize=18, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Volume subplot
        ax2.bar(price_data.index, price_data['volume'], 
               color='gray', alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('trade_setup.png', dpi=300, bbox_inches='tight')
        plt.close()

        return 'trade_setup.png'

# Create the visualizer
visualizer = TheoryVisualizer()

# Example usage
print("Visualization module created with methods:")
print("- plot_wyckoff_schematic()")
print("- plot_smc_structures()")
print("- plot_confluence_heatmap()")
print("- plot_trade_setup()")
