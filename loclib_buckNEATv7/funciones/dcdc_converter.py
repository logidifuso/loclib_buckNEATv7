#
# Implementación de la evaluación del fitness para el convertidor dcdc
#
import math as m
import numpy as np
import matplotlib.pyplot as plt
import neat
import warnings


class BuckClass:
    """
    Contiene los métodos para:
    - Obtener el método a usar para evaluación del fitness (en función del modelo de
     Buck utilizado y del uso o no de multiprocessing
    - Obtener el método a usar para graficar la salida de la red
    - Métodos de evaluación del fitness para las funciones a aprender
    - Métodos para graficar la salida producida por la ANN (y la función objetivo)

    Los modelos implementados son:
    - buck_l1a      : Modelo buck nuevo, r_c = 0
    - buck_l1a_pid  : Modelo buck nuevo, r_c = 0, PID annadido
    """

    def __init__(self, dcdc_config):

        self.modelo = dcdc_config['modelo']

        # Componentes del convertidor
        self.l_in = dcdc_config['l_in']
        self.l_out = dcdc_config['l_out']
        self.c_in = dcdc_config['c_in']
        self.c_out = dcdc_config['c_out']
        self.res_hs = dcdc_config['res_hs']
        self.res_ls = dcdc_config['res_ls']
        self.r_l = dcdc_config['r_l']
        self.r_c = dcdc_config['r_c']

        # Condiciones de operación y evaluación
        self.T = dcdc_config['period']
        self.target_vout = dcdc_config['target_vout']
        self.penalty = dcdc_config['penalty']
        self.tolerancia = dcdc_config['tolerancia']
        self.tsettling_tol = dcdc_config['t_settling_tol']
        self.t_exp_penalty = dcdc_config['t_settling_penalty']
        self.fit_a_coef = dcdc_config['fit_a_coef']
        self.fit_b_coef = dcdc_config['fit_b_coef']

        # Estado inicial
        self.i_lout = dcdc_config['i_lout_0']
        self.v_out = dcdc_config['v_out_0']
        self.v_ix = dcdc_config['v_ix_0']
        self.i_li = dcdc_config['i_li_0']
        self.duty = dcdc_config['duty_0']

        # Definición test_sequence
        self.type_vin = dcdc_config['type_vin']
        self.type_rload = dcdc_config['type_rload']
        self.vals_vin = dcdc_config['vals_vin']
        self.vals_rload = dcdc_config['vals_rload']

        # Tiempo simulado
        self.steps = int(dcdc_config['steps'])
        self.steady = int(dcdc_config['steady'])

        # Coeficientes del controlador PID
        self.kp = dcdc_config['kp']
        self.ki = dcdc_config['ki']
        self.kd = dcdc_config['kd']
        self.error = np.zeros(self.steps + self.steady)

        # -----------------------------------------------------------------------
        # Atributos de entradas a manejar - decide funciones a utilizar para
        # generar las secuencias de Vin y Rload
        # -----------------------------------------------------------------------
        do1 = f"func_sequence_{self.type_vin}"
        func_sequence_vin = getattr(self, do1)

        do2 = f"func_sequence_{self.type_rload}"
        func_sequence_rload = getattr(self, do2)

        # -----------------------------------------------------------------------
        # Valores instantes tiempo para cambios en el escalón de carga.
        # Duración de los pulsos de carga y vector t multiplicador para "itxae"
        # -----------------------------------------------------------------------
        self.t_pulse = int(self.steps / 4)
        self.rise_edge1 = self.steady
        self.fall_edge1 = self.rise_edge1 + self.t_pulse
        self.rise_edge2 = self.fall_edge1 + self.t_pulse
        self.fall_edge2 = self.rise_edge2 + self.t_pulse

        self.t_factor = (np.arange(0, self.t_pulse, 1)) / self.tsettling_tol
        selecc_t_factor = np.greater(self.t_factor, 1)
        self.t_factor[selecc_t_factor] = self.t_factor[selecc_t_factor]**self.t_exp_penalty


        # ------------------------------------------------------------------------
        # Generación de secuencias Vin y Rload
        # -----------------------------------------------------------------------
        self.sequence_vin = func_sequence_vin(self.vals_vin)
        self.sequence_rload = func_sequence_rload(self.vals_rload)

        # -----------------------------------------------------------------------
        # Coeficientes para simulación en buck_status_update según valores de los
        # componentes
        # -----------------------------------------------------------------------
        self.a11 = -self.r_l / self.l_out
        self.a12 = -1 / self.l_out
        self.a21 = (self.l_out - self.c_out * self.r_l * self.r_c) / (self.l_out * self.c_out)
        self.a22a = self.r_c * self.c_out
        self.a22b = self.l_out * self.c_out
        self.b2 = self.r_c / self.l_out

#   -----------------------------------------------------------------------------------
#   Creacción de secuencias de Vin y Rload. Staticmethods llamados desde __init__
#   -----------------------------------------------------------------------------------
    def func_sequence_constante(self, vals):
        steps = self.steps + self.steady
        ampl = vals[0]
        secuencia = ampl * np.ones(steps)
        return secuencia

    def func_sequence_seno(self, vals):
        """
        Unidades de tiempo se asumen en µsegundos.
        Duración de la secuencia: 10000 pasos <==> 10000 µs => 100 Hz
        :param vals:
        :return:
        """
        nominal_val = vals[0]
        ampl = vals[1]
        freq = vals[2]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val
        for i in range(self.steps):
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2*m.pi*freq*(i/10**6))
        return secuencia

    def func_sequence_barridof(self, vals):
        nominal_val = vals[0]
        ampl = vals[1]
        f_inic = vals[2]
        f_fin = vals[3]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val

        slope = (f_fin - f_inic) / self.steps

        for i in range(self.steps):
            freq = f_inic + slope * i
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2 * m.pi * freq * (i / 10 ** 6))
        return secuencia

    def func_sequence_full_sweep(self, vals):
        nominal_val = vals[0]
        ampl = vals[1]
        f_inic = vals[2]
        f_fin = vals[3]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val

        for i in range(self.steps):
            freq = f_inic * 10**(np.log10(f_fin/f_inic)*(i/self.steps))
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2 * m.pi * freq * (i / 10 ** 6))
        return secuencia

    def func_sequence_barridof_inverso(self, vals):
        nominal_val = vals[0]
        ampl = vals[1]
        f_inic = vals[2]
        f_fin = vals[3]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val

        for i in range(self.steps):
            freq = f_inic * np.exp(np.log(f_fin/f_inic) * (i/self.steps)**5 )
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2 * m.pi * freq * (i / 10 ** 6))
        return secuencia


    def func_sequence_escalon(self, vals):
        min_val = vals[0]
        max_val = vals[1]

        secuencia = np.empty(self.steps + self.steady)

        secuencia[:self.rise_edge1] = max_val
        secuencia[self.rise_edge1:self.fall_edge1] = min_val
        secuencia[self.fall_edge1:self.rise_edge2] = max_val
        secuencia[self.rise_edge2:self.fall_edge2] = min_val
        secuencia[self.fall_edge2:] = max_val

        return secuencia


#   -----------------------------------------------------------------------------------
#   Métodos de simulación y evaluación del fitness
#   -----------------------------------------------------------------------------------
    def devuelve_metodo_eval(self, mp):
        """
        Devuelve el método a utilizar para la evaluación del fitness en función del modelo
        :param mp: bool multiprocessing
        :return: retorna como objeto el método a utilizar para evaluar fitness
        """
        x = self.modelo
        if mp:
            do = f"eval_genomes_mp_{x}"
            if hasattr(self, do) and callable(getattr(self, do)):
                func = getattr(self, do)
                return func
        else:
            do = f"eval_genomes_single_{x}"
            if hasattr(self, do) and callable(getattr(self, do)):
                func = getattr(self, do)
                return func

    def devuelve_fitness_eval(self):
        """
        Devuelve el método para obtener el fitness de un genoma según el modelo
        :return:
        """
        x = self.modelo
        do = f"fitness_{x}"
        if hasattr(self, do) and callable(getattr(self, do)):
            func = getattr(self, do)
            return func

    # --------------------------------------------------------------------------------- #
    # -------------                Level 1 - l2_5i                 -------------------- #
    # --------------------------------------------------------------------------------- #

    def eval_genomes_mp_buck_l2_5i(self, genomes, config):

        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_l2_5i(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_l2_5i(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_l2_5i(self, net)

    def fitness_buck_l2_5i(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        vout = self.run_buck_simulation_l2_5i(net)[1]
        error = np.absolute(vout - self.target_vout)
        error[0:self.steady] = 0
        # Normalización del error respecto a la tolerancia especificada
        error = error / self.tolerancia
        # Selección de componentes por encima del error y penalización adicional (exponente)
        selecc = np.greater(error, 1)
        error[selecc] = error[selecc]**self.penalty

        iaxe = error.sum() / self.steps

        itxae1 = np.sum(error[self.rise_edge1:self.fall_edge1] * self.t_factor)
        itxae2 = np.sum(error[self.fall_edge1:self.rise_edge2] * self.t_factor)
        itxae3 = np.sum(error[self.rise_edge2:self.fall_edge2] * self.t_factor)
        itxae4 = np.sum(error[self.fall_edge2:self.fall_edge2+self.t_pulse] * self.t_factor)

        itxae_tot = itxae1 + itxae2 + itxae3 + itxae4
        error_tot = self.fit_a_coef * iaxe + self.fit_b_coef * itxae_tot
        return np.exp(-error_tot)


    def run_buck_simulation_l2_5i(self, net):

        steps = self.steps + self.steady

        i_lout_record = np.zeros(steps)
        vout_record = np.ones(steps) * self.v_out
        duty_record = np.zeros(steps+1)

        u_i = 0
        # Ejecuta la simulación
        for i in range(2, steps-1):
            duty_record[i] = self.duty

            # Aplica la salida de RN y obtiene nuevo estado. En realidad no hace falta
            # pasar self.duty, puedo leerlo dentro de buck_status_update pero por claridad...
            self.buck_status_update_l1a(self.sequence_vin[i],
                                        self.sequence_rload[i],
                                        self.duty)

            # Activa la RN y obtiene su salida
            output_ann = net.activate([self.sequence_rload[i],
                                       self.v_out,
                                       vout_record[i-1],  # ])[0]
                                       vout_record[i-2],
                                       self.i_lout])[0]   # self.x2,

            # TODO: Ojo!! si sigo sumando 0.5, hay que cambiar la función de activación a tanh
            self.duty = 0.5 + output_ann
            self.duty = min(max(self.duty, 0.01), 0.99)

            # Record estado
            i_lout_record[i] = self.i_lout
            vout_record[i] = self.v_out

        return i_lout_record, vout_record, duty_record

    # Graficado de los resultados
    def plot_respuesta_buck_l2_5i(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        simul_results = self.run_buck_simulation_l2_5i(net)
        self.plot_respuesta_buck_l1(simul_results, tinic, tfinal, view, filename)


    # --------------------------------------------------------------------------------- #
    # -------------------------- Level 1a - l1a_5i_pid------------------------------------ #
    # --------------------------------------------------------------------------------- #

    def eval_genomes_mp_buck_l2_5i_pid(self, genomes, config):

        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_l2_5i_pid(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_l2_5i_pid(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_l2_5i_pid(self, net)

    def fitness_buck_l2_5i_pid(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        vout = self.run_buck_simulation_l2_5i_pid(net)[1]
        error = (vout - self.target_vout)
        error[0:self.steady] = 0
        error = np.exp(self.penalty * np.absolute(error)) - 1
        error_tot = error.sum() / self.steps
        return np.exp(-error_tot)

    def run_buck_simulation_l2_5i_pid(self, net):

        steps = self.steps + self.steady

        i_lout_record = np.zeros(steps)
        vout_record = np.ones(steps) * self.v_out
        duty_record = np.zeros(steps+1)

        u_i = 0
        # Ejecuta la simulación
        for i in range(2, steps-1):
            duty_record[i] = self.duty

            # Aplica la salida de RN y obtiene nuevo estado. En realidad no hace falta
            # pasar self.duty, puedo leerlo dentro de buck_status_update pero por claridad...
            self.buck_status_update_l1a(self.sequence_vin[i],
                                        self.sequence_rload[i],
                                        self.duty)

            # Activa la RN y obtiene su salida
            output_ann = net.activate([self.sequence_rload[i],
                                       self.v_out,
                                       vout_record[i-1],  # ])[0]
                                       vout_record[i-2],
                                       self.i_lout])[0]   # self.x2,

            self.error[i] = self.target_vout - self.v_out
            u_p = self.kp * self.error[i]
            u_i = u_i + self.ki * self.error[i]
            u_i = min(max(u_i, 0), 1)
            u_d = self.kd * (self.error[i] - self.error[i-1])

            # TODO: Ojo, que si sigo sumando 0.5 hay que usar tanh !!!
            self.duty = u_p + u_i + u_d + 0.5 + output_ann

            self.duty = min(max(self.duty, 0.01), 0.99)
            # Record estado
            i_lout_record[i] = self.i_lout
            vout_record[i] = self.v_out

        return i_lout_record, vout_record, duty_record

    # Graficado de los resultados
    def plot_respuesta_buck_l2_5i_pid(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        simul_results = self.run_buck_simulation_l2_5i_pid(net)
        self.plot_respuesta_buck_l1(simul_results, tinic, tfinal, view, filename)



    # ==============================================================================================
    #               Funciones de uso general por los diferentes "sub-modelos"
    # ==============================================================================================

    # buck_status_update equiv. a do_step
    def buck_status_update_l1a(self, v_in, r_load, duty):
        """
        Versión de la función basada en el modelo simplificado del paper ...x
        :param v_in:
        :param r_load:
        :param duty:
        :return:
        """
        i_lout_new = self.i_lout + self.T * (self.a11 * self.i_lout + self.a12 * self.v_out +
                                             (v_in * duty)/self.l_out)

        v_out_new = self.v_out + self.T * (self.i_lout/self.c_out - self.v_out/(r_load*self.c_out))

        self.i_lout = i_lout_new
        self.v_out = v_out_new

    # --------------------- Graficas ---------------------------------------
    def devuelve_metodo_graf(self):
        x = self.modelo
        do = f"plot_respuesta_{x}"
        if hasattr(self, do) and callable(getattr(self, do)):
            func = getattr(self, do)
            return func

    # Graficado de los resultados
    def plot_respuesta_buck_l1(self, simul_results,
                               tinic=0, tfinal=None, view=False, filename='salida.svg'):
        """
        :param simul_results
        :param tinic:
        :param tfinal:
        :param view:
        :param filename:
        :return:
        """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        if tfinal is None:
            tfinal = len(simul_results[0])

        simul_lenght = len(simul_results[0])
        t = np.linspace(0, simul_lenght-1, simul_lenght)

        fig, axs = plt.subplots(4, figsize=(18, 18))
        fig.suptitle('Evolución temporal')

        axs[0].plot(t[tinic:tfinal], self.sequence_rload[tinic:tfinal])
        axs[0].set(xlabel='time (µs)', ylabel='Current (A)', title='Rload')

        axs[1].plot(t[tinic:tfinal], simul_results[0][tinic:tfinal])
        axs[1].set(xlabel='time (µs)', ylabel='Current (A)', title='i_lout')

        axs[2].plot(t[tinic:tfinal], simul_results[1][tinic:tfinal])
        axs[2].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vout')

        axs[3].plot(t[tinic:tfinal], simul_results[2][tinic:tfinal])
        axs[3].set(xlabel='time (µs)', ylabel='duty', title='PWM duty')

        for i in range(4):
            for j in range(2):
                axs[i].grid(True)

        plt.savefig(filename)
        if view:
            plt.show()
        plt.close()
