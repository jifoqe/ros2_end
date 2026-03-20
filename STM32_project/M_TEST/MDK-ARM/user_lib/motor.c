#include "motor.h"


/* foot position define */
/*-----------------------------------------------------*/
extern TIM_HandleTypeDef htim1;
extern GPIO_InitTypeDef GPIO_InitStruct;
#define M_TIM htim1
#define M1_SPEED TIM_CHANNEL_1 
#define M2_SPEED TIM_CHANNEL_2 
#define M3_SPEED TIM_CHANNEL_3 
#define M4_SPEED TIM_CHANNEL_4 
/*-----------------------------------------------------*/
/* end */

/* motor function overview */
/*-----------------------------------------------------*/

void Motor_Init(void);
void Motor_Speed_Percent(uint8_t device, uint8_t percent, int8_t direction);

/*-----------------------------------------------------*/
/* end */


/* This funtion is used for init the motor */
/*-----------------------------------------------------*/

void Motor_Init(void)
{
		HAL_TIM_PWM_Start(&M_TIM, M1_SPEED);
		HAL_TIM_PWM_Start(&M_TIM, M2_SPEED);
		HAL_TIM_PWM_Start(&M_TIM, M3_SPEED);
		HAL_TIM_PWM_Start(&M_TIM, M4_SPEED);
		Motor_Speed_Percent(1, 0, 0);
		Motor_Speed_Percent(2, 0, 0);
		Motor_Speed_Percent(3, 0, 0);
		Motor_Speed_Percent(4, 0, 0);
}

/*-----------------------------------------------------*/
/* end */

/* This funtion is used to control the motor by a percentage */
/*-----------------------------------------------------*/

void	Motor_Speed_Percent(uint8_t device, uint8_t percent, int8_t direction)
{
	if( percent >100)
		percent = 100;
	
	/* percent to ccr */
	uint32_t motor_ccr = (7199 * percent) / 100;
	
	/* determine the direction */
	switch(device)
	{
		case 1:
			__HAL_TIM_SET_COMPARE(&M_TIM, M1_SPEED, motor_ccr);
			if( direction == 1)
			{
				HAL_GPIO_WritePin(M1_I1_GPIO_Port, M1_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M1_I2_GPIO_Port, M1_I2_Pin, GPIO_PIN_SET);
				
			}
			else if( direction == -1)
			{
				HAL_GPIO_WritePin(M1_I1_GPIO_Port, M1_I1_Pin, GPIO_PIN_SET);
				HAL_GPIO_WritePin(M1_I2_GPIO_Port, M1_I2_Pin, GPIO_PIN_RESET);
			}
			else
			{
				HAL_GPIO_WritePin(M1_I1_GPIO_Port, M1_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M1_I2_GPIO_Port, M1_I2_Pin, GPIO_PIN_RESET);
			}
			break;
		case 2:
			__HAL_TIM_SET_COMPARE(&M_TIM, M2_SPEED, motor_ccr);
			if( direction == 1)
			{
				HAL_GPIO_WritePin(M2_I1_GPIO_Port, M2_I1_Pin, GPIO_PIN_SET);
				HAL_GPIO_WritePin(M2_I2_GPIO_Port, M2_I2_Pin, GPIO_PIN_RESET);
				
			}
			else if( direction == -1)
			{
				HAL_GPIO_WritePin(M2_I1_GPIO_Port, M2_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M2_I2_GPIO_Port, M2_I2_Pin, GPIO_PIN_SET);
			}
			else
			{
				HAL_GPIO_WritePin(M2_I1_GPIO_Port, M2_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M2_I2_GPIO_Port, M2_I2_Pin, GPIO_PIN_RESET);
			}
			break;
		case 3:
			__HAL_TIM_SET_COMPARE(&M_TIM, M3_SPEED, motor_ccr);
			if( direction == 1)
			{
				HAL_GPIO_WritePin(M3_I1_GPIO_Port, M3_I1_Pin, GPIO_PIN_SET);
				HAL_GPIO_WritePin(M3_I2_GPIO_Port, M3_I2_Pin, GPIO_PIN_RESET);
				
			}
			else if( direction == -1)
			{
				HAL_GPIO_WritePin(M3_I1_GPIO_Port, M3_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M3_I2_GPIO_Port, M3_I2_Pin, GPIO_PIN_SET);
			}
			else
			{
				HAL_GPIO_WritePin(M3_I1_GPIO_Port, M3_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M3_I2_GPIO_Port, M3_I2_Pin, GPIO_PIN_RESET);
			}
			break;
		case 4:
			__HAL_TIM_SET_COMPARE(&M_TIM, M4_SPEED, motor_ccr);
			if( direction == 1)
			{
				HAL_GPIO_WritePin(M4_I1_GPIO_Port, M4_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M4_I2_GPIO_Port, M4_I2_Pin, GPIO_PIN_SET);
				
			}
			else if( direction == -1)
			{
				HAL_GPIO_WritePin(M4_I1_GPIO_Port, M4_I1_Pin, GPIO_PIN_SET);
				HAL_GPIO_WritePin(M4_I2_GPIO_Port, M4_I2_Pin, GPIO_PIN_RESET);
			}
			else
			{
				HAL_GPIO_WritePin(M4_I1_GPIO_Port, M4_I1_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(M4_I2_GPIO_Port, M4_I2_Pin, GPIO_PIN_RESET);
			}
			break;
		default:
			__HAL_TIM_SET_COMPARE(&M_TIM, M1_SPEED, 0);
			__HAL_TIM_SET_COMPARE(&M_TIM, M2_SPEED, 0);
			__HAL_TIM_SET_COMPARE(&M_TIM, M3_SPEED, 0);
			__HAL_TIM_SET_COMPARE(&M_TIM, M4_SPEED, 0);
			break;
	}
}

/*-----------------------------------------------------*/

